"""
Inference script for DiT with block-wise spatial cosine similarity analysis.

This script:
1. Loads a trained DiT checkpoint
2. Generates samples using flow-based sampling
3. Computes block-wise cosine similarity at specified timesteps
4. Logs similarity matrices to WandB

Usage:
    python inference_similarity.py --checkpoint_path /path/to/checkpoint.pkl \\
                                   --num_samples 8 \\
                                   --cfg_scale 3.0
"""

import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import flax
import wandb
from tqdm import tqdm

from diffusion_transformer import DiT
from utils.stable_vae import StableVAE
from utils.checkpoint import Checkpoint


def compute_block_cosine_matrix(block_tokens):
    """
    Compute pairwise cosine similarity matrix between all blocks.

    Args:
        block_tokens: List of L tensors, each with shape (B, N, D)
                      where L = num_blocks, B = batch_size,
                      N = num_tokens, D = hidden_dim

    Returns:
        sim_mat: (L, L) cosine similarity matrix averaged over batch and tokens
    """
    # Stack into single tensor: (L, B, N, D)
    H = jnp.stack(block_tokens, axis=0)
    L, B, N, D = H.shape

    # Normalize along hidden dimension
    H_norm = H / (jnp.linalg.norm(H, axis=-1, keepdims=True) + 1e-8)

    # Compute pairwise cosine: (L, 1, B, N, D) * (1, L, B, N, D) -> (L, L, B, N, D)
    # Then sum over D to get cosine values
    cosine_vals = jnp.einsum('labnd,lbbnd->labn', H_norm[:, None], H_norm[None, :])

    # Average over batch and tokens: (L, L, B, N) -> (L, L)
    sim_mat = jnp.mean(cosine_vals, axis=(-2, -1))

    return sim_mat


def sample_with_similarity_tracking(model, vae_decode, initial_latent, labels,
                                    denoise_steps=50, cfg_scale=3.0,
                                    track_timesteps=[1, 4, 8, 16, 32, 63]):
    """
    Sample from DiT while tracking block-wise cosine similarity at specified timesteps.

    Args:
        model: FlaxTrainer instance with DiT model
        vae_decode: VAE decode function
        initial_latent: (B, H, W, C) initial noise
        labels: (B,) class labels
        denoise_steps: number of denoising steps
        cfg_scale: classifier-free guidance scale
        track_timesteps: list of timestep indices to track similarity

    Returns:
        final_images: (B, H, W, 3) generated images
        similarity_data: dict with timestep -> sim_mat mapping
    """
    x = initial_latent
    B = x.shape[0]
    similarity_data = {}

    delta_t = 1.0 / denoise_steps

    for step_idx in tqdm(range(denoise_steps), desc="Sampling"):
        # Current timestep (using midpoint for better integration)
        t_val = (step_idx + 0.5) / denoise_steps
        t_vec = jnp.full((B,), t_val)

        # CFG: create conditional and unconditional inputs
        num_classes = model.config['num_classes']
        labels_uncond = jnp.ones_like(labels) * num_classes

        x_expanded = jnp.tile(x, (2, 1, 1, 1))  # (2B, H, W, C)
        t_expanded = jnp.tile(t_vec, (2,))       # (2B,)
        labels_full = jnp.concatenate([labels, labels_uncond], axis=0)  # (2B,)

        # Forward pass with block tokens
        if step_idx in track_timesteps:
            # Get block tokens for similarity analysis
            v_pred, block_tokens = model.model_eps(
                x_expanded, t_expanded, labels_full,
                train=False, force_drop_ids=False, return_block_tokens=True
            )
        else:
            # Normal forward without block tokens
            v_pred = model.model_eps(
                x_expanded, t_expanded, labels_full,
                train=False, force_drop_ids=False
            )

        # Split conditional and unconditional predictions
        v_cond = v_pred[:B]
        v_uncond = v_pred[B:]

        # Apply CFG
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        # Update latent using Euler integration
        x = x + v * delta_t

        # Compute and store similarity matrix if tracking this timestep
        if step_idx in track_timesteps:
            # Extract conditional branch block tokens only
            block_tokens_cond = [bt[:B] for bt in block_tokens]

            # Compute cosine similarity matrix
            sim_mat = compute_block_cosine_matrix(block_tokens_cond)

            # Store (convert to numpy for easier handling)
            similarity_data[step_idx] = np.array(sim_mat)

            print(f"  [Step {step_idx}] Computed similarity matrix, shape: {sim_mat.shape}")

    # Decode final latent to image
    x_decoded = vae_decode(x)
    final_images = np.array(jnp.clip(x_decoded * 0.5 + 0.5, 0, 1))

    return final_images, similarity_data


def main(args):
    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        name=f"inference_similarity_{args.run_name}",
        config=vars(args)
    )

    print("="*80)
    print("DiT Block-wise Similarity Inference")
    print("="*80)

    # Setup JAX
    print(f"JAX devices: {jax.local_devices()}")
    device_count = len(jax.local_devices())

    # Load VAE
    print("\nLoading Stable VAE...")
    vae = StableVAE.create()
    vae_decode = jax.jit(vae.decode)

    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint_path}")
    cp = Checkpoint(args.checkpoint_path)

    # Create model config (need to match training config)
    model_config = {
        'hidden_size': args.hidden_size,
        'patch_size': args.patch_size,
        'depth': args.depth,
        'num_heads': args.num_heads,
        'mlp_ratio': 4,
        'class_dropout_prob': 0.15,
        'num_classes': args.num_classes,
        'use_gram_branch': args.use_gram_branch,
        'gram_rank': args.gram_rank,
    }

    # Initialize model
    print("\nInitializing DiT model...")
    print(f"  Config: {model_config}")

    # Create dummy trainer to hold model
    # (We reuse FlowTrainer structure but only for inference)
    from train_flow import FlowTrainer
    from utils.train_state import TrainState
    import optax

    rng = jax.random.PRNGKey(args.seed)
    rng, param_key, dropout_key = jax.random.split(rng, 3)

    # Create model
    dit_args = {k: model_config[k] for k in
                ['patch_size', 'hidden_size', 'depth', 'num_heads', 'mlp_ratio',
                 'class_dropout_prob', 'num_classes', 'use_gram_branch', 'gram_rank']}
    dit_args['debug'] = False
    model_def = DiT(**dit_args)

    # Initialize with dummy input
    example_latent = jnp.zeros((1, args.latent_size, args.latent_size, args.latent_channels))
    example_t = jnp.zeros((1,))
    example_label = jnp.zeros((1,), dtype=jnp.int32)
    params = model_def.init(
        {'params': param_key, 'label_dropout': dropout_key},
        example_latent, example_t, example_label
    )['params']

    # Create trainer
    tx = optax.adam(learning_rate=0.0001)  # dummy optimizer
    model_ts = TrainState.create(model_def, params, tx=tx)
    model_ts_eps = TrainState.create(model_def, params)
    trainer = FlowTrainer(rng, model_ts, model_ts_eps, model_config)

    # Load checkpoint weights
    trainer = cp.load_model(trainer)
    print(f"Loaded model at step: {trainer.model.step}")

    # Generate samples
    print(f"\n{'='*80}")
    print(f"Generating {args.num_samples} samples...")
    print(f"  CFG scale: {args.cfg_scale}")
    print(f"  Denoising steps: {args.denoise_steps}")
    print(f"  Tracking timesteps: {args.track_timesteps}")
    print(f"{'='*80}\n")

    # Create initial noise
    rng, noise_key = jax.random.split(rng)
    initial_latent = jax.random.normal(
        noise_key,
        (args.num_samples, args.latent_size, args.latent_size, args.latent_channels)
    )

    # Sample class labels
    rng, label_key = jax.random.split(rng)
    labels = jax.random.randint(label_key, (args.num_samples,), 0, args.num_classes)

    # Run sampling with similarity tracking
    final_images, similarity_data = sample_with_similarity_tracking(
        trainer,
        vae_decode,
        initial_latent,
        labels,
        denoise_steps=args.denoise_steps,
        cfg_scale=args.cfg_scale,
        track_timesteps=args.track_timesteps
    )

    print(f"\nGenerated {len(final_images)} images")
    print(f"Similarity data collected at {len(similarity_data)} timesteps")

    # Log to WandB
    print("\nLogging to WandB...")

    # Log generated images
    wandb_images = [wandb.Image(img, caption=f"Class {labels[i]}")
                   for i, img in enumerate(final_images)]
    wandb.log({"generated_samples": wandb_images})

    # Log similarity matrices
    for timestep, sim_mat in similarity_data.items():
        # Log as heatmap
        fig = wandb.plot.heatmap(
            x_labels=[f"Block {i}" for i in range(sim_mat.shape[0])],
            y_labels=[f"Block {i}" for i in range(sim_mat.shape[1])],
            matrix_values=sim_mat.tolist(),
            show_text=True
        )
        wandb.log({f"similarity/timestep_{timestep:03d}": fig})

        # Also log raw matrix
        wandb.log({f"similarity_matrix/timestep_{timestep:03d}": wandb.Table(
            data=sim_mat.tolist(),
            columns=[f"Block_{i}" for i in range(sim_mat.shape[1])]
        )})

    # Compute and log summary statistics
    avg_sim_mat = np.mean(list(similarity_data.values()), axis=0)
    wandb.log({
        "similarity/average_across_timesteps": wandb.plot.heatmap(
            x_labels=[f"Block {i}" for i in range(avg_sim_mat.shape[0])],
            y_labels=[f"Block {i}" for i in range(avg_sim_mat.shape[1])],
            matrix_values=avg_sim_mat.tolist(),
            show_text=True
        )
    })

    print("\n✅ Inference complete!")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiT inference with block similarity analysis")

    # Model config
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--use_gram_branch", type=bool, default=True)
    parser.add_argument("--gram_rank", type=int, default=64)

    # Sampling config
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--latent_size", type=int, default=28)
    parser.add_argument("--latent_channels", type=int, default=4)

    # Similarity tracking config
    parser.add_argument("--track_timesteps", type=int, nargs="+",
                       default=[1, 4, 8, 16, 32, 63],
                       help="Timestep indices to track similarity")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="dit-similarity")
    parser.add_argument("--run_name", type=str, default="test")

    args = parser.parse_args()
    main(args)
