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
flags.DEFINE_string('fid_stats', None, 'FID stats file.')
flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval interval.')
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
    'use_gram_branch': False,  # Set to True to enable Gram branch
    'gram_rank': 64,           # Low-rank dimension for Gram branch
    # Logging
    'loss_ema_beta': 0.99,     # EMA smoothing factor for loss
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

def get_x_t(images, eps, t):
    t = jnp.clip(t, 0, 1 - 0.01)
    return (1 - t) * eps + t * images

def get_v(images, eps):
    return images - eps

class FlowTrainer(flax.struct.PyTreeNode):
    rng: Any
    model: TrainState
    model_eps: TrainState
    config: dict = flax.struct.field(pytree_node=False)
    loss_ema: float = 0.0  # EMA of training loss

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
        setup_wandb(FLAGS.model.to_dict(), **FLAGS.wandb)

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
    dit_args  = {k: FLAGS.model[k] for k in ['patch_size','hidden_size','depth','num_heads','mlp_ratio','class_dropout_prob','num_classes','use_gram_branch','gram_rank']}
    model_def = DiT(**dit_args)

    example_t     = jnp.zeros((1,))
    example_label = jnp.zeros((1,), dtype=jnp.int32)
    params = model_def.init({'params': param_key, 'label_dropout': dropout_key}, example_obs, example_t, example_label)['params']
    print("Total num of parameters:", sum(x.size for x in jax.tree_util.tree_leaves(params)))

    tx           = optax.adam(learning_rate=FLAGS.model['lr'], b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'])
    model_ts     = TrainState.create(model_def, params, tx=tx)
    model_ts_eps = TrainState.create(model_def, params)
    model        = FlowTrainer(rng, model_ts, model_ts_eps, FLAGS.model)

    if FLAGS.load_dir is not None:
        cp    = Checkpoint(FLAGS.load_dir)
        model = cp.load_model(model)
        print("Loaded model with step", model.model.step)
        del cp

    if FLAGS.fid_stats is not None:
        from utils.fid import get_fid_network, fid_from_stats
        get_fid_activations = get_fid_network()
        truth_fid_stats     = np.load(FLAGS.fid_stats)

    model = flax.jax_utils.replicate(model, devices=jax.local_devices())
    model = model.replace(rng=jax.random.split(rng, len(jax.local_devices())))
    jax.debug.visualize_array_sharding(model.model.params['FinalLayer_0']['Dense_0']['bias'])

    valid_images_small, _ = next(dataset_valid)
    valid_images_small    = valid_images_small[:device_count, None]
    visualize_labels      = example_labels.reshape((device_count, -1, *example_labels.shape[1:]))[:, 0:1]
    if FLAGS.model.use_stable_vae:
        valid_images_small = vae_encode_pmap(vae_rng, valid_images_small)

    ###################################
    # Train Loop
    ###################################
    def process_img(img):
        if FLAGS.model.use_stable_vae:
            img = vae_decode(img[None])[0]
        return np.array(jnp.clip(img * 0.5 + 0.5, 0, 1))

    def eval_model():
        valid_images, valid_labels = next(dataset_valid)
        valid_images = valid_images.reshape((device_count, -1, *valid_images.shape[1:]))
        valid_labels = valid_labels.reshape((device_count, -1, *valid_labels.shape[1:]))
        if FLAGS.model.use_stable_vae:
            valid_images = vae_encode_pmap(vae_rng, valid_images)

        _, valid_info = model.update(valid_images, valid_labels)
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
            }

            if jax.process_index() == 0:
                wandb.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0 or i == 1000:
            eval_model()

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            if jax.process_index() == 0:
                model_single = flax.jax_utils.unreplicate(model)
                cp = Checkpoint(FLAGS.save_dir, parallel=False)
                cp.set_model(model_single)
                cp.save()
            del cp, model_single

if __name__ == '__main__':
    app.run(main)