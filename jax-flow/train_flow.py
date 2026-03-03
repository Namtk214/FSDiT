# from localutils.debugger import enable_debug
# enable_debug()

from typing import Any
import os
import time
from absl import app, flags
from functools import partial
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import flax
import optax
import wandb
from ml_collections import config_flags
import ml_collections
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")
import matplotlib.pyplot as plt

from utils.wandb_utils import setup_wandb, default_wandb_config
from utils.train_state import TrainState, target_update
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from diffusion_transformer import DiT

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/kaggle/input/datasets/arjunashok33/miniimagenet', 'Dataset directory.')
flags.DEFINE_string('load_dir', None, 'Load checkpoint dir.')
flags.DEFINE_string('save_dir', None, 'Save checkpoint dir.')
flags.DEFINE_string('fid_stats', None, 'FID stats file. If None, will auto-generate from validation set.')
flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('fid_interval', 50000, 'FID evaluation interval. Set to 0 to disable.')
flags.DEFINE_integer('fid_num_samples', 1000, 'Number of samples to generate for FID.')
flags.DEFINE_integer('save_interval', 200000, 'Save interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1_000_000), 'Number of training steps.')
flags.DEFINE_integer('debug_overfit', 0, 'Debug overfitting.')

model_config = ml_collections.ConfigDict({
    'lr': 0.0001,
    'beta1': 0.9,
    'beta2': 0.99,
    'hidden_size': 768,
    'patch_size': 2,
    'depth': 12,
    'num_heads': 12,
    'mlp_ratio': 4,
    'class_dropout_prob': 0.1,
    'num_classes': 100,
    'denoise_timesteps': 32,
    'cfg_scale': 4.0,
    'target_update_rate': 0.9999,
    't_sampler': 'log-normal',
    't_conditioning': 1,
    'preset': 'big',
    'use_stable_vae': 1,
    # Gram branch support (new)
    'use_gram_branch': True,   # Set to True to enable Gram branch
    'gram_rank': 64,           # Low-rank dimension for Gram branch
    # Logging
    'loss_ema_beta': 0.99,     # EMA smoothing factor for loss
    # Debug
    'debug_model': False,      # Enable debug logging in DiT model
})

preset_configs = {
    'debug':     {'hidden_size': 64,   'patch_size': 8, 'depth': 2,  'num_heads': 2,  'mlp_ratio': 1},
    'big':       {'hidden_size': 768,  'patch_size': 2, 'depth': 12, 'num_heads': 12, 'mlp_ratio': 4},
    'semilarge': {'hidden_size': 1024, 'patch_size': 2, 'depth': 22, 'num_heads': 16, 'mlp_ratio': 4},
    'large':     {'hidden_size': 1024, 'patch_size': 2, 'depth': 24, 'num_heads': 16, 'mlp_ratio': 4},
    'xlarge':    {'hidden_size': 1152, 'patch_size': 2, 'depth': 28, 'num_heads': 16, 'mlp_ratio': 4},
}

wandb_config = default_wandb_config()
wandb_config.update({'project': 'flow', 'name': 'flow_miniimagenet'})
config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('model', model_config, lock_config=False)

##############################################
## Model Definitions.
##############################################

def estimate_dit_flops_per_step(batch_size, image_size, patch_size, hidden_size, depth, num_heads, mlp_ratio, in_channels=4, use_gram_branch=False, gram_rank=64):
    """
    Estimate FLOPs per training step (forward + backward + optimizer) for DiT model.

    Args:
        batch_size: Training batch size
        image_size: Input image size (H=W)
        patch_size: Patch size
        hidden_size: Hidden dimension D
        depth: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        in_channels: Number of input channels
        use_gram_branch: Whether Gram branch is enabled
        gram_rank: Low-rank dimension r for Gram branch

    Returns:
        F_step: Estimated FLOPs per training step
    """
    N = (image_size // patch_size) ** 2  # Number of patches
    D = hidden_size

    # Patch embedding: Conv2d with kernel=patch_size, stride=patch_size
    # FLOPs = 2 * output_size * kernel_ops
    # output_size = N (number of patches)
    # kernel_ops = patch_size * patch_size * in_channels * D
    patch_embed_flops = 2 * batch_size * N * (patch_size * patch_size * in_channels * D)

    # Timestep embedding: 2 Dense layers (256 -> D -> D)
    timestep_flops = 2 * batch_size * (2 * 256 * D + 2 * D * D)

    # Label/Support embedding: Embedding lookup or MLP (assume MLP with 2 layers)
    label_flops = 2 * batch_size * (2 * D * D * 2)  # Approximate

    # Per DiTBlock:
    per_block_flops = 0

    # 1. Attention path
    # - LayerNorm: ~2ND (negligible, skip)
    # - Modulation MLP (c -> 6D): 2 * batch_size * D * 6D
    per_block_flops += 2 * batch_size * D * (6 * D)

    # - QKV projection: 3 separate Dense(D -> D) on N tokens
    per_block_flops += 2 * batch_size * N * 3 * D * D

    # - Attention computation: Q @ K^T: (B, H, N, d) @ (B, H, d, N) -> (B, H, N, N)
    #   FLOPs = B * H * N * N * d * 2
    head_dim = D // num_heads
    per_block_flops += 2 * batch_size * num_heads * N * N * head_dim

    # - Softmax: ~5N^2 per head (approximate)
    per_block_flops += 5 * batch_size * num_heads * N * N

    # - Attention @ V: (B, H, N, N) @ (B, H, N, d) -> (B, H, N, d)
    per_block_flops += 2 * batch_size * num_heads * N * N * head_dim

    # - Output projection: Dense(D -> D) on N tokens
    per_block_flops += 2 * batch_size * N * D * D

    # 2. MLP path
    # - LayerNorm: negligible
    # - Modulation: already counted above (6D output includes both gates)
    # - MLP: Dense(D -> mlp_ratio*D) + Dense(mlp_ratio*D -> D)
    mlp_dim = int(mlp_ratio * D)
    per_block_flops += 2 * batch_size * N * (D * mlp_dim + mlp_dim * D)

    # 3. Gram branch (ADDITIVE if enabled)
    if use_gram_branch:
        r = gram_rank
        # G = X · X_t^T: (B, N, D) @ (B, D, N) → (B, N, N)
        gram_matrix_flops = 2 * batch_size * N * N * D

        # Rg = G · A · B:
        # - G @ A: (B, N, N) @ (N, r) → (B, N, r)
        # - (G@A) @ B: (B, N, r) @ (r, D) → (B, N, D)
        gram_proj_flops = 2 * batch_size * N * N * r + 2 * batch_size * N * r * D

        # RMSNorm: ~B*N*D (negligible compared to matmul)
        gram_norm_flops = batch_size * N * D

        # Total Gram FLOPs per block
        gram_flops_per_block = gram_matrix_flops + gram_proj_flops + gram_norm_flops
        per_block_flops += gram_flops_per_block

    # Total for all blocks
    transformer_flops = depth * per_block_flops

    # Final layer
    # - LayerNorm + Modulation: 2 * batch_size * D * 2D
    # - Dense(D -> patch_size^2 * in_channels) on N tokens
    out_dim = patch_size * patch_size * in_channels
    final_flops = 2 * batch_size * D * (2 * D) + 2 * batch_size * N * D * out_dim

    # Total forward FLOPs
    F_fwd = patch_embed_flops + timestep_flops + label_flops + transformer_flops + final_flops

    # Training step: forward + backward (2x forward) + optimizer (~0.2x forward)
    # Use conservative estimate: F_step = 3 * F_fwd
    F_step = 3.0 * F_fwd

    return F_step

def get_x_t(images, eps, t):
    t = jnp.clip(t, 0, 1 - 0.01)
    return (1 - t) * eps + t * images

def get_v(images, eps):
    return images - eps

def create_fid_stats_from_dataset(data_dir, is_train, num_samples, device_count,
                                   vae_encode_pmap, vae_rng, vae_decode_pmap,
                                   use_stable_vae, get_fid_activations):
    """
    Create FID statistics from validation dataset (creates separate iterator).

    Args:
        data_dir: Dataset directory
        is_train: Whether to use train or val split
        num_samples: Number of samples to use for computing stats
        device_count: Number of devices for pmap
        vae_encode_pmap: VAE encode function (if using stable VAE)
        vae_rng: Random key for VAE
        vae_decode_pmap: VAE decode function (pmap version)
        use_stable_vae: Whether using stable VAE
        get_fid_activations: InceptionV3 feature extraction function

    Returns:
        (mu, sigma): Mean and covariance of InceptionV3 features
    """
    print(f"Creating FID stats from {num_samples} validation samples...")

    # Create temporary dataset iterator for FID stats
    split = 'train' if is_train else 'val'
    ds_dir = os.path.join(data_dir, split)
    dataset = tf.keras.utils.image_dataset_from_directory(
        ds_dir, image_size=(224, 224), batch_size=None,
        label_mode='int', shuffle=False, seed=42,
    )
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - 0.5) / 0.5
        return image, tf.cast(label, tf.int32)
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(device_count * 16, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset_iter = iter(dataset.as_numpy_iterator())

    all_activations = []
    samples_collected = 0

    while samples_collected < num_samples:
        try:
            batch_images, _ = next(dataset_iter)
        except StopIteration:
            break

        batch_size = batch_images.shape[0]
        # Pad to device_count if needed
        if batch_size % device_count != 0:
            pad_size = device_count - (batch_size % device_count)
            batch_images = np.concatenate([batch_images, batch_images[:pad_size]], axis=0)

        batch_images = batch_images.reshape((device_count, -1, *batch_images.shape[1:]))

        # Encode and decode if using VAE
        if use_stable_vae:
            batch_images = vae_encode_pmap(vae_rng, batch_images)
            batch_images = vae_decode_pmap(batch_images)

        # Convert to [0, 1]
        batch_images = jnp.clip(batch_images * 0.5 + 0.5, 0, 1)

        # Resize to 256x256 for InceptionV3
        batch_images_flat = np.array(batch_images).reshape(-1, *batch_images.shape[2:])
        if batch_images_flat.shape[1] != 256:
            batch_images_tf = tf.constant(batch_images_flat)
            batch_images_resized = tf.image.resize(batch_images_tf, [256, 256], method='bilinear')
            batch_images_flat = np.array(batch_images_resized)

        # Convert to [-1, 1] for InceptionV3
        batch_images_flat = batch_images_flat * 2.0 - 1.0

        # Extract features
        batch_images_pmap = batch_images_flat.reshape((device_count, -1, 256, 256, 3))
        activations = get_fid_activations(batch_images_pmap)
        activations = np.array(activations).reshape(-1, 2048)

        # Collect activations
        remaining = num_samples - samples_collected
        activations = activations[:min(len(activations), remaining)]
        all_activations.append(activations)
        samples_collected += len(activations)

        if samples_collected % 500 == 0 or samples_collected >= num_samples:
            print(f"  Collected {samples_collected}/{num_samples} samples for FID stats")

    # Concatenate and compute statistics
    all_activations = np.concatenate(all_activations, axis=0)[:num_samples]
    mu = np.mean(all_activations, axis=0)
    sigma = np.cov(all_activations, rowvar=False)

    print(f"FID stats created: mu shape {mu.shape}, sigma shape {sigma.shape}")
    return mu, sigma

class FlowTrainer(flax.struct.PyTreeNode):
    rng: Any
    model: TrainState
    model_eps: TrainState
    config: dict = flax.struct.field(pytree_node=False)
    loss_ema: float = 0.0  # EMA of training loss

    @partial(jax.pmap, axis_name='data')
    def evaluate(self, images, labels, pmap_axis='data'):
        """Evaluate model WITHOUT updating params (for validation)"""
        new_rng, label_key, time_key, noise_key = jax.random.split(self.rng, 4)

        # Same loss computation as training, but NO gradient/update
        if self.config['t_sampler'] == 'log-normal':
            t = jax.random.normal(time_key, (images.shape[0],))
            t = 1 / (1 + jnp.exp(-t))
        else:
            t = jax.random.uniform(time_key, (images.shape[0],))
        t_full = t[:, None, None, None]
        eps = jax.random.normal(noise_key, images.shape)
        x_t = get_x_t(images, eps, t_full)
        v_t = get_v(images, eps)
        if self.config['t_conditioning'] == 0:
            t = jnp.zeros_like(t)

        # Use EMA model for evaluation (NOT training model)
        v_prime = self.model_eps(x_t, t, labels, train=False, force_drop_ids=False)
        loss = jnp.mean((v_prime - v_t) ** 2)

        info = {
            'l2_loss': loss,
            'v_abs_mean': jnp.abs(v_t).mean(),
            'v_pred_abs_mean': jnp.abs(v_prime).mean(),
        }
        info = jax.lax.pmean(info, axis_name=pmap_axis)

        # Return info only, NO model update
        return info

    @partial(jax.pmap, axis_name='data')
    def update(self, images, labels, pmap_axis='data'):
        new_rng, label_key, time_key, noise_key = jax.random.split(self.rng, 4)

        def loss_fn(params):
            if self.config['t_sampler'] == 'log-normal':
                t = jax.random.normal(time_key, (images.shape[0],))
                t = 1 / (1 + jnp.exp(-t))
            else:
                t = jax.random.uniform(time_key, (images.shape[0],))
            t_full = t[:, None, None, None]
            eps = jax.random.normal(noise_key, images.shape)
            x_t = get_x_t(images, eps, t_full)
            v_t = get_v(images, eps)
            if self.config['t_conditioning'] == 0:
                t = jnp.zeros_like(t)
            v_prime = self.model(x_t, t, labels, train=True, rngs={'label_dropout': label_key}, params=params)
            loss = jnp.mean((v_prime - v_t) ** 2)
            return loss, {
                'l2_loss': loss,
                'v_abs_mean': jnp.abs(v_t).mean(),
                'v_pred_abs_mean': jnp.abs(v_prime).mean(),
            }

        grads, info = jax.grad(loss_fn, has_aux=True)(self.model.params)
        grads = jax.lax.pmean(grads, axis_name=pmap_axis)
        info = jax.lax.pmean(info, axis_name=pmap_axis)

        updates, new_opt_state = self.model.tx.update(grads, self.model.opt_state, self.model.params)
        new_params = optax.apply_updates(self.model.params, updates)
        new_model = self.model.replace(step=self.model.step + 1, params=new_params, opt_state=new_opt_state)

        info['grad_norm'] = optax.global_norm(grads)
        info['update_norm'] = optax.global_norm(updates)
        info['param_norm'] = optax.global_norm(new_params)

        # Update EMA loss
        beta = self.config.get('loss_ema_beta', 0.99)
        # Use jnp.where instead of if to avoid tracer error in pmap
        new_loss_ema = jnp.where(
            self.model.step == 0,
            info['l2_loss'],  # First step: use current loss
            beta * self.loss_ema + (1 - beta) * info['l2_loss']  # Otherwise: EMA update
        )
        info['loss_ema'] = new_loss_ema

        # Log learning rate (extract from optimizer state)
        info['lr'] = self.config['lr']

        # Update the model_eps
        new_model_eps = target_update(self.model, self.model_eps, 1-self.config['target_update_rate'])
        if self.config['target_update_rate'] == 1:
            new_model_eps = new_model
        new_trainer = self.replace(rng=new_rng, model=new_model, model_eps=new_model_eps, loss_ema=new_loss_ema)
        return new_trainer, info


    @partial(jax.jit, static_argnames=('cfg'))
    def call_model(self, images, t, labels, cfg=True, cfg_val=1.0):
        if self.config['t_conditioning'] == 0:
            t = jnp.zeros_like(t)
        if not cfg:
            return self.model_eps(images, t, labels, train=False, force_drop_ids=False)
        else:
            labels_uncond = jnp.ones(labels.shape, dtype=jnp.int32) * self.config['num_classes'] # Null token
            images_expanded = jnp.tile(images, (2, 1, 1, 1)) # (batch*2, h, w, c)
            t_expanded = jnp.tile(t, (2,)) # (batch*2,)
            labels_full = jnp.concatenate([labels, labels_uncond], axis=0)
            v_pred = self.model_eps(images_expanded, t_expanded, labels_full, train=False, force_drop_ids=False)
            v_label = v_pred[:images.shape[0]]
            v_uncond = v_pred[images.shape[0]:]
            v = v_uncond + cfg_val * (v_label - v_uncond)
            return v

    @partial(jax.pmap, axis_name='data', in_axes=(0, 0, 0, 0), static_broadcasted_argnums=(4,5))
    def call_model_pmap(self, images, t, labels, cfg=True, cfg_val=1.0):
        return self.call_model(images, t, labels, cfg=cfg, cfg_val=cfg_val)

##############################################
## Training Code.
##############################################
def main(_):

    preset_dict = preset_configs[FLAGS.model.preset]
    for k, v in preset_dict.items():
        FLAGS.model[k] = v

    np.random.seed(FLAGS.seed)
    print("Using devices", jax.local_devices())
    device_count = len(jax.local_devices())
    global_device_count = jax.device_count()
    print("Device count", device_count)
    print("Global device count", global_device_count)
    local_batch_size = FLAGS.batch_size // (global_device_count // device_count)
    print("Global Batch: ", FLAGS.batch_size)
    print("Node Batch: ", local_batch_size)
    print("Device Batch:", local_batch_size // device_count)

    # Create wandb logger
    if jax.process_index() == 0:
        wandb_config = FLAGS.model.to_dict()
        setup_wandb(wandb_config, **FLAGS.wandb)

    def get_dataset(is_train):
        split   = 'train' if (is_train or FLAGS.debug_overfit) else 'val'
        ds_dir  = os.path.join(FLAGS.data_dir, split)
        dataset = tf.keras.utils.image_dataset_from_directory(
            ds_dir, image_size=(224, 224), batch_size=None,
            label_mode='int', shuffle=is_train, seed=42,
        )
        def preprocess(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            if is_train:
                image = tf.image.random_flip_left_right(image)
            image = (image - 0.5) / 0.5
            return image, tf.cast(label, tf.int32)
        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if FLAGS.debug_overfit:
            dataset = dataset.take(8)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
        dataset = dataset.batch(local_batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return iter(dataset.as_numpy_iterator())

    dataset       = get_dataset(is_train=True)
    dataset_valid = get_dataset(is_train=False)
    example_obs, example_labels = next(dataset)
    example_obs = example_obs[:1]

    if FLAGS.model.use_stable_vae:
        vae             = StableVAE.create()
        example_obs     = vae.encode(jax.random.PRNGKey(0), example_obs)
        vae_rng         = flax.jax_utils.replicate(jax.random.PRNGKey(42))
        vae_encode_pmap = jax.pmap(vae.encode)
        vae_decode      = jax.jit(vae.decode)
        vae_decode_pmap = jax.pmap(vae.decode)

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, param_key, dropout_key = jax.random.split(rng, 3)
    print("Total Memory on device:", float(jax.local_devices()[0].memory_stats()['bytes_limit']) / 1024**3, "GB")

    FLAGS.model.image_channels = example_obs.shape[-1]
    FLAGS.model.image_size     = example_obs.shape[1]
    dit_args  = {k: FLAGS.model[k] for k in ['patch_size','hidden_size','depth','num_heads','mlp_ratio','class_dropout_prob','num_classes','use_gram_branch','gram_rank','debug_model']}
    # Rename debug_model to debug for DiT
    dit_args['debug'] = dit_args.pop('debug_model')
    model_def = DiT(**dit_args)

    example_t     = jnp.zeros((1,))
    example_label = jnp.zeros((1,), dtype=jnp.int32)
    params = model_def.init({'params': param_key, 'label_dropout': dropout_key}, example_obs, example_t, example_label)['params']
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    print("\n" + "="*80)
    print("📊 MODEL INITIALIZATION")
    print("="*80)
    print(f"Total parameters: {total_params:,}")

    # Detailed param breakdown
    print(f"\n📊 PARAMETER BREAKDOWN:")

    # Count attention params (should ALWAYS exist)
    attn_params = 0
    for i in range(FLAGS.model.depth):
        block_key = f'DiTBlock_{i}'
        if block_key in params:
            if 'MultiHeadDotProductAttention_0' in params[block_key]:
                attn_block = params[block_key]['MultiHeadDotProductAttention_0']
                for k, v in attn_block.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            if hasattr(vv, 'size'):
                                attn_params += vv.size
                    elif hasattr(v, 'size'):
                        attn_params += v.size

    if attn_params > 0:
        print(f"✓ Attention params: {attn_params:,} ({attn_params/total_params*100:.2f}% of model)")
    else:
        print(f"⚠️  WARNING: No attention params found!")

    # Count Gram params (only if enabled)
    if FLAGS.model.use_gram_branch:
        gram_params = 0
        for i in range(FLAGS.model.depth):
            block_key = f'DiTBlock_{i}'
            if block_key in params:
                if 'gram_A' in params[block_key] and 'gram_B' in params[block_key]:
                    gram_A_size = params[block_key]['gram_A'].size
                    gram_B_size = params[block_key]['gram_B'].size
                    gram_params += gram_A_size + gram_B_size
                    if i == 0:  # Print first block details
                        print(f"\n✓ Gram branch (ADDITIVE):")
                        print(f"  - Rank: {FLAGS.model.gram_rank}")
                        print(f"  - Gram_A shape: {params[block_key]['gram_A'].shape} ({gram_A_size:,} params)")
                        print(f"  - Gram_B shape: {params[block_key]['gram_B'].shape} ({gram_B_size:,} params)")

        if gram_params > 0:
            print(f"  - Total Gram params: {gram_params:,} ({gram_params/total_params*100:.2f}% of model)")
            print(f"\n🔵 GRAM-DiT MODE: Attention + Gram branch (both active)")
        else:
            print(f"\n⚠️  WARNING: use_gram_branch=True but no Gram params found!")
    else:
        print(f"\n❌ Standard DiT: Attention only (no Gram branch)")

    print("="*80 + "\n")

    # Estimate FLOPs per training step
    F_step = estimate_dit_flops_per_step(
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.model.image_size,
        patch_size=FLAGS.model.patch_size,
        hidden_size=FLAGS.model.hidden_size,
        depth=FLAGS.model.depth,
        num_heads=FLAGS.model.num_heads,
        mlp_ratio=FLAGS.model.mlp_ratio,
        in_channels=FLAGS.model.image_channels,
        use_gram_branch=FLAGS.model.use_gram_branch,
        gram_rank=FLAGS.model.gram_rank
    )
    print(f"Estimated FLOPs per step: {F_step:.2e}")
    print(f"Estimated TFLOPs per step: {F_step / 1e12:.4f}")

    if FLAGS.model.use_gram_branch:
        # Estimate Gram overhead
        N = (FLAGS.model.image_size // FLAGS.model.patch_size) ** 2
        D = FLAGS.model.hidden_size
        r = FLAGS.model.gram_rank
        gram_flops_per_block = FLAGS.batch_size * N * (2*N*D + 2*N*r + 2*r*D + D)
        gram_total = gram_flops_per_block * FLAGS.model.depth * 3.0  # forward+backward+opt
        print(f"Gram branch overhead: {gram_total/1e12:.4f} TFLOPs ({gram_total/F_step*100:.1f}% of total)")

    tx           = optax.adam(learning_rate=FLAGS.model['lr'], b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'])
    model_ts     = TrainState.create(model_def, params, tx=tx)
    model_ts_eps = TrainState.create(model_def, params)
    model        = FlowTrainer(rng, model_ts, model_ts_eps, FLAGS.model)

    if FLAGS.load_dir is not None:
        cp    = Checkpoint(FLAGS.load_dir)
        model = cp.load_model(model)
        print("Loaded model with step", model.model.step)
        del cp

    # Setup FID evaluation
    fid_enabled = FLAGS.fid_interval > 0
    if fid_enabled:
        from utils.fid import get_fid_network, fid_from_stats
        get_fid_activations = get_fid_network()

        # Load or create FID stats
        if FLAGS.fid_stats is not None and os.path.exists(FLAGS.fid_stats):
            print(f"Loading FID stats from {FLAGS.fid_stats}")
            truth_fid_stats = np.load(FLAGS.fid_stats)
            fid_mu_real = truth_fid_stats['mu']
            fid_sigma_real = truth_fid_stats['sigma']
        else:
            print("FID stats not found. Will create from validation set...")
            fid_mu_real = None
            fid_sigma_real = None

    # Log model info to wandb
    if jax.process_index() == 0:
        wandb.log({
            'model/num_params': total_params,
            'model/trainable_params': total_params,
            'model/estimated_tflops_per_step': F_step / 1e12,
        }, step=0)

    model = flax.jax_utils.replicate(model, devices=jax.local_devices())
    model = model.replace(rng=jax.random.split(rng, len(jax.local_devices())))
    jax.debug.visualize_array_sharding(model.model.params['FinalLayer_0']['Dense_0']['bias'])

    # Create FID stats if needed (only on process 0)
    if fid_enabled and fid_mu_real is None and jax.process_index() == 0:
        print("Creating FID reference statistics from validation set...")
        fid_mu_real, fid_sigma_real = create_fid_stats_from_dataset(
            data_dir=FLAGS.data_dir,
            is_train=False,
            num_samples=5000,
            device_count=device_count,
            vae_encode_pmap=vae_encode_pmap if FLAGS.model.use_stable_vae else None,
            vae_rng=vae_rng if FLAGS.model.use_stable_vae else None,
            vae_decode_pmap=vae_decode_pmap if FLAGS.model.use_stable_vae else None,
            use_stable_vae=FLAGS.model.use_stable_vae,
            get_fid_activations=get_fid_activations
        )
        # Save for future use
        if FLAGS.save_dir is not None:
            fid_stats_path = os.path.join(FLAGS.save_dir, 'fid_stats.npz')
            os.makedirs(FLAGS.save_dir, exist_ok=True)
            np.savez(fid_stats_path, mu=fid_mu_real, sigma=fid_sigma_real)
            print(f"Saved FID stats to {fid_stats_path}")
    elif fid_enabled and fid_mu_real is None:
        # Wait for process 0 to create stats
        fid_stats_path = os.path.join(FLAGS.save_dir, 'fid_stats.npz') if FLAGS.save_dir else None
        if fid_stats_path:
            while not os.path.exists(fid_stats_path):
                time.sleep(5)
            truth_fid_stats = np.load(fid_stats_path)
            fid_mu_real = truth_fid_stats['mu']
            fid_sigma_real = truth_fid_stats['sigma']
            print(f"Loaded FID stats from {fid_stats_path}")

    valid_images_small, _ = next(dataset_valid)
    valid_images_small    = valid_images_small[:device_count, None]
    visualize_labels      = example_labels.reshape((device_count, -1, *example_labels.shape[1:]))[:, 0:1]
    if FLAGS.model.use_stable_vae:
        valid_images_small = vae_encode_pmap(vae_rng, valid_images_small)

    ###################################
    # FID Evaluation Function
    ###################################
    def compute_fid(step):
        """Generate samples and compute FID score."""
        if not fid_enabled or jax.process_index() != 0:
            return None

        print(f"[Step {step}] Computing FID...")
        num_fid_samples = FLAGS.fid_num_samples
        samples_per_device = (num_fid_samples + device_count - 1) // device_count
        samples_per_batch = min(32, samples_per_device)  # Generate in smaller batches
        num_batches = (samples_per_device + samples_per_batch - 1) // samples_per_batch

        all_generated = []
        for batch_idx in range(num_batches):
            batch_start = batch_idx * samples_per_batch
            batch_end = min((batch_idx + 1) * samples_per_batch, samples_per_device)
            actual_batch_size = batch_end - batch_start

            # Sample random labels for generation
            fid_labels = np.random.randint(0, FLAGS.model.num_classes, size=(device_count, actual_batch_size))

            # Generate from noise
            key = jax.random.PRNGKey(42 + batch_idx + step)
            if FLAGS.model.use_stable_vae:
                noise_shape = (device_count, actual_batch_size, FLAGS.model.image_size, FLAGS.model.image_size, FLAGS.model.image_channels)
            else:
                noise_shape = (device_count, actual_batch_size, 224, 224, 3)
            eps = jax.random.normal(key, noise_shape)

            # Denoise with CFG
            x = eps
            delta_t = 1.0 / FLAGS.model.denoise_timesteps
            for ti in range(FLAGS.model.denoise_timesteps):
                t_vec = jnp.full((x.shape[0], x.shape[1]), ti / FLAGS.model.denoise_timesteps)
                x = x + model.call_model_pmap(x, t_vec, fid_labels, True, FLAGS.model.cfg_scale) * delta_t

            # Decode if using VAE
            if FLAGS.model.use_stable_vae:
                x = vae_decode_pmap(x)

            # Convert to numpy and clip to [0, 1]
            x = np.array(x)
            x = np.clip(x * 0.5 + 0.5, 0, 1)

            # Flatten device dimension and collect
            x = x.reshape(-1, *x.shape[2:])
            all_generated.append(x)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Generated {(batch_idx + 1) * samples_per_batch * device_count}/{num_fid_samples} samples")

        # Concatenate all generated samples
        generated_samples = np.concatenate(all_generated, axis=0)[:num_fid_samples]

        # Resize to 256x256 for InceptionV3
        if generated_samples.shape[1] != 256:
            generated_samples_tf = tf.constant(generated_samples)
            generated_samples_resized = tf.image.resize(generated_samples_tf, [256, 256], method='bilinear')
            generated_samples = np.array(generated_samples_resized)

        # Convert to [-1, 1] for InceptionV3
        generated_samples = generated_samples * 2.0 - 1.0

        # Extract features using InceptionV3
        generated_samples_pmap = generated_samples.reshape((device_count, -1, 256, 256, 3))
        activations = get_fid_activations(generated_samples_pmap)
        activations = np.array(activations).reshape(-1, 2048)[:num_fid_samples]

        # Compute statistics
        mu_gen = np.mean(activations, axis=0)
        sigma_gen = np.cov(activations, rowvar=False)

        # Compute FID
        fid_score = fid_from_stats(fid_mu_real, fid_sigma_real, mu_gen, sigma_gen)

        print(f"[Step {step}] FID score: {fid_score:.2f}")
        return float(fid_score)

    ###################################
    # Train Loop
    ###################################
    def process_img(img):
        if FLAGS.model.use_stable_vae:
            img = vae_decode(img[None])[0]
        return np.array(jnp.clip(img * 0.5 + 0.5, 0, 1))

    def eval_model():
        # Validation loss
        valid_images, valid_labels = next(dataset_valid)
        valid_images = valid_images.reshape((device_count, -1, *valid_images.shape[1:]))
        valid_labels = valid_labels.reshape((device_count, -1, *valid_labels.shape[1:]))
        if FLAGS.model.use_stable_vae:
            valid_images = vae_encode_pmap(vae_rng, valid_images)

        # Use evaluate() instead of update() - NO param update!
        valid_info = model.evaluate(valid_images, valid_labels)
        if jax.process_index() == 0:
            valid_info_np = jax.tree.map(lambda x: float(np.array(x).mean()), valid_info)
            val_metrics = {
                'val/loss': valid_info_np['l2_loss'],
                'val/v_abs_mean': valid_info_np['v_abs_mean'],
                'val/v_pred_abs_mean': valid_info_np['v_pred_abs_mean'],
            }
            wandb.log(val_metrics, step=i)

        # One-step denoising visualization (8 devices only)
        if len(jax.local_devices()) == 8:
            key = jax.random.PRNGKey(42)
            t   = jnp.repeat(jnp.arange(8)[:, None] / 8, valid_images.shape[1], axis=1)
            eps = jax.random.normal(key, valid_images.shape)
            x_t = get_x_t(valid_images, eps, t[..., None, None, None])
            x_1_pred = x_t + model.call_model_pmap(x_t, t, valid_labels, False, 0.0) * (1 - t[..., None, None, None])
            if jax.process_index() == 0:
                fig, axs = plt.subplots(8, 24, figsize=(90, 30))
                for j in range(8):
                    for k in range(8):
                        axs[j, 3*k].imshow(process_img(valid_images[j, k]),  vmin=0, vmax=1)
                        axs[j, 3*k+1].imshow(process_img(x_t[j, k]),        vmin=0, vmax=1)
                        axs[j, 3*k+2].imshow(process_img(x_1_pred[j, k]),   vmin=0, vmax=1)
                wandb.log({'reconstruction': wandb.Image(fig)}, step=i)
                plt.close(fig)

        # Full denoising at various CFG
        key     = jax.random.PRNGKey(42 + jax.process_index() + i)
        eps     = jax.random.normal(key, valid_images_small.shape)
        delta_t = 1.0 / FLAGS.model.denoise_timesteps
        for cfg_scale in [0, 1, 4]:
            x = eps
            for ti in range(FLAGS.model.denoise_timesteps):
                t_vec = jnp.full((x.shape[0], x.shape[1]), ti / FLAGS.model.denoise_timesteps)
                x = x + model.call_model_pmap(x, t_vec, visualize_labels, True, cfg_scale) * delta_t
            if jax.process_index() == 0:
                fig, axs = plt.subplots(1, device_count, figsize=(30, 5))
                for j in range(device_count):
                    axs[j].imshow(process_img(np.array(x)[j, 0]), vmin=0, vmax=1)
                    axs[j].set_title(f"class {visualize_labels[j, 0]}")
                wandb.log({f'sample_cfg_{cfg_scale}': wandb.Image(fig)}, step=i)
                plt.close(fig)

        del valid_images, valid_labels
        print("Finished eval")

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, dynamic_ncols=True):
        step_start_time = time.time()

        if not FLAGS.debug_overfit or i == 1:
            batch_images, batch_labels = next(dataset)
            batch_images = batch_images.reshape((device_count, -1, *batch_images.shape[1:]))
            batch_labels = batch_labels.reshape((device_count, -1, *batch_labels.shape[1:]))
            if FLAGS.model.use_stable_vae:
                batch_images = vae_encode_pmap(vae_rng, batch_images)

        model, update_info = model.update(batch_images, batch_labels)

        step_time = time.time() - step_start_time

        if i % FLAGS.log_interval == 0:
            update_info = jax.tree.map(lambda x: np.array(x), update_info)
            update_info = jax.tree.map(lambda x: x.mean(), update_info)

            # Calculate TFLOPS
            tflops = F_step / (step_time * 1e12)

            # Build comprehensive training metrics
            train_metrics = {
                'train/loss': update_info['l2_loss'],
                'train/loss_ema': update_info['loss_ema'],
                'train/lr': update_info['lr'],
                'train/grad_norm': update_info['grad_norm'],
                'train/param_norm': update_info['param_norm'],
                'train/v_abs_mean': update_info['v_abs_mean'],
                'train/v_pred_abs_mean': update_info['v_pred_abs_mean'],
                'perf/train_step_time': step_time,
                'perf/tflops': tflops,
            }

            if jax.process_index() == 0:
                wandb.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0 or i == 1000:
            eval_model()

        # FID evaluation at separate interval
        if fid_enabled and i % FLAGS.fid_interval == 0 and i > 0:
            fid_score = compute_fid(i)
            if fid_score is not None and jax.process_index() == 0:
                wandb.log({'eval/FID': fid_score}, step=i)

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            if jax.process_index() == 0:
                model_single = flax.jax_utils.unreplicate(model)
                save_target = FLAGS.save_dir
                save_ext = os.path.splitext(os.path.basename(save_target))[1]
                if save_target.endswith("/") or not save_ext:
                    os.makedirs(save_target, exist_ok=True)
                    save_target = os.path.join(save_target, f"checkpoint_step_{i}.pkl")
                cp = Checkpoint(save_target, parallel=False)
                cp.set_model(model_single)
                cp.save()
                del cp, model_single

if __name__ == '__main__':
    app.run(main)
