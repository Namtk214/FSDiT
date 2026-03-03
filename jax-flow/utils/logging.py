"""
utils/logging.py — Logging helpers for FSDiT training.

Extracts metric computation and wandb logging from train.py for cleaner
separation. All functions are pure: they take data, compute metrics,
and log to wandb.
"""

import numpy as np

try:
    import wandb
except ImportError:
    wandb = None


# ═══════════════════════════════════════════════════════════════════════════════
#  Metric computation (pure numpy, no wandb dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_condition_distribution_metrics(cond_vec, class_ids):
    """
    Compute SigLIP pooled-condition distribution metrics for a batch.

    Args:
        cond_vec: (B, 768) float32 pooled support embeddings.
        class_ids: (B,) int32 class IDs.

    Returns:
        metrics: dict of scalar metrics
        same_vals: cosine values for same-class pairs
        diff_vals: cosine values for diff-class pairs
    """
    x = np.asarray(cond_vec, dtype=np.float32)
    y = np.asarray(class_ids, dtype=np.int32).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        return {}, np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    bsz = x.shape[0]
    metrics = {
        'cond/support_pooled_abs_mean': float(np.mean(np.abs(x))),
        'cond/support_pooled_l2_mean': float(np.mean(np.linalg.norm(x, axis=-1))),
        'cond/support_pooled_dim_std_mean': float(np.mean(np.std(x, axis=0))),
    }

    if bsz < 2:
        metrics['cond/same_class_pair_ratio'] = 0.0
        metrics['cond/same_class_cos_mean'] = 0.0
        metrics['cond/same_class_cos_std'] = 0.0
        metrics['cond/diff_class_cos_mean'] = 0.0
        metrics['cond/diff_class_cos_std'] = 0.0
        return metrics, np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    x_norm = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)
    cos = x_norm @ x_norm.T
    same = (y[:, None] == y[None, :])
    iu = np.triu_indices(bsz, 1)
    cos_u = cos[iu]
    same_u = same[iu]
    same_vals = cos_u[same_u]
    diff_vals = cos_u[~same_u]
    num_pairs = float(cos_u.shape[0])
    metrics['cond/same_class_pair_ratio'] = float(same_vals.shape[0] / max(num_pairs, 1.0))
    metrics['cond/same_class_cos_mean'] = float(np.mean(same_vals)) if same_vals.size else 0.0
    metrics['cond/same_class_cos_std'] = float(np.std(same_vals)) if same_vals.size else 0.0
    metrics['cond/diff_class_cos_mean'] = float(np.mean(diff_vals)) if diff_vals.size else 0.0
    metrics['cond/diff_class_cos_std'] = float(np.std(diff_vals)) if diff_vals.size else 0.0
    return metrics, same_vals.astype(np.float32), diff_vals.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  wandb log builders
# ═══════════════════════════════════════════════════════════════════════════════

def log_train_metrics(
    step, info, loss_ema, lr_schedule, cfg, dt_step=0.0,
    sup_pooled_global=None, class_ids_global=None,
    log_model_debug=True, cond_hist_interval=5000,
):
    """
    Log training metrics to wandb.

    Args:
        step: Current training step.
        info: Dict returned by Trainer.train_step (contains pmap-ed arrays).
        loss_ema: Mutable [float] container for EMA loss.
        lr_schedule: optax learning rate schedule callable.
        cfg: Model config dict.
        dt_step: Total wall-clock time for this iteration.
        sup_pooled_global: (B, 768) support embeddings for condition metrics.
        class_ids_global: (B,) class IDs for condition metrics.
        log_model_debug: Whether to log model debug metrics.
        cond_hist_interval: Steps between condition histogram logging.
    """
    alpha = cfg.loss_ema_alpha
    train_loss = float(np.array(info['loss']).mean())
    if loss_ema[0] is None:
        loss_ema[0] = train_loss
    else:
        loss_ema[0] = alpha * loss_ema[0] + (1 - alpha) * train_loss

    log = {
        'train/loss': train_loss,
        'train/loss_ema': loss_ema[0],
        'train/grad_norm': float(np.array(info['grad_norm']).mean()),
        'train/lr': float(lr_schedule(step)),
        'train/step_time': dt_step,
        'train/param_norm': float(np.array(info['param_norm']).mean()),
    }
    tbin = np.array(info['tbin_loss']).mean(axis=0)
    for b in range(cfg.num_t_bins):
        log[f'train/loss_tbin_{b}'] = float(tbin[b])

    if log_model_debug:
        for key in ('t_emb_abs_mean', 'y_emb_abs_mean', 'c_abs_mean', 'c_l2_mean',
                     'support_pooled_abs_mean', 'support_pooled_l2_mean'):
            dbg_key = f'dbg/{key}'
            if dbg_key in info:
                log[dbg_key] = float(np.array(info[dbg_key]).mean())
                # Rename support_pooled keys for clarity
                if 'support_pooled' in key:
                    log[f'dbg/{key}_model'] = log.pop(dbg_key)

        if 'dbg/act_abs_per_layer' in info:
            act_abs = np.array(info['dbg/act_abs_per_layer']).mean(axis=0)
            act_rms = np.array(info['dbg/act_rms_per_layer']).mean(axis=0)
            for li in range(cfg.depth):
                log[f'act/layer{li}_abs_mean'] = float(act_abs[li])
                log[f'act/layer{li}_rms'] = float(act_rms[li])

    if sup_pooled_global is not None and class_ids_global is not None:
        cond_metrics, same_vals, diff_vals = compute_condition_distribution_metrics(
            sup_pooled_global, class_ids_global
        )
        log.update(cond_metrics)
        if step % cond_hist_interval == 0:
            log['cond/support_pooled_hist'] = wandb.Histogram(
                sup_pooled_global.reshape(-1)
            )
            if same_vals.size:
                log['cond/same_class_cos_hist'] = wandb.Histogram(same_vals)
            if diff_vals.size:
                log['cond/diff_class_cos_hist'] = wandb.Histogram(diff_vals)

    wandb.log(log, step=step)


def log_perf_metrics(step, data_time, siglip_time, vae_time, step_time,
                     total_time, data_mode='grain', siglip_stats=None):
    """Log performance / timing metrics to wandb."""
    perf_log = {
        'perf/data_time': data_time,
        'perf/siglip_encode_time': siglip_time,
        'perf/vae_time': vae_time,
        'perf/train_step_time': step_time,
        'perf/total_iter_time': total_time,
    }
    if data_mode == 'online' and siglip_stats is not None:
        perf_log['perf/siglip_cache_hit_rate'] = float(siglip_stats['cache_hit_rate'])
        perf_log['perf/siglip_cache_items'] = float(siglip_stats['cache_items'])
        perf_log['perf/siglip_unique_paths_per_batch'] = float(
            siglip_stats['unique_paths_per_batch']
        )
    wandb.log(perf_log, step=step)


def log_eval_metrics(step, v_loss_f, v_tbin_f, loss_ema, cfg,
                     val_sup_pooled_global, val_class_ids,
                     data_mode='grain', val_siglip_stats=None,
                     cond_hist_interval=5000):
    """Log evaluation / validation metrics to wandb."""
    gap = (loss_ema[0] - v_loss_f) if loss_ema[0] else 0.0
    log = {'val/loss': v_loss_f, 'val/train_val_gap': gap}
    for b in range(cfg.num_t_bins):
        log[f'val/loss_tbin_{b}'] = float(v_tbin_f[b])

    cond_metrics, same_vals, diff_vals = compute_condition_distribution_metrics(
        val_sup_pooled_global, val_class_ids
    )
    for k, v in cond_metrics.items():
        log[f'val_{k}'] = v

    if data_mode == 'online' and val_siglip_stats is not None:
        log['val_perf/siglip_encode_time'] = float(val_siglip_stats['encode_time'])
        log['val_perf/siglip_cache_hit_rate'] = float(val_siglip_stats['cache_hit_rate'])
        log['val_perf/siglip_cache_items'] = float(val_siglip_stats['cache_items'])
        log['val_perf/siglip_unique_paths_per_batch'] = float(
            val_siglip_stats['unique_paths_per_batch']
        )

    if step % cond_hist_interval == 0:
        log['val_cond/support_pooled_hist'] = wandb.Histogram(
            val_sup_pooled_global.reshape(-1)
        )
        if same_vals.size:
            log['val_cond/same_class_cos_hist'] = wandb.Histogram(same_vals)
        if diff_vals.size:
            log['val_cond/diff_class_cos_hist'] = wandb.Histogram(diff_vals)

    wandb.log(log, step=step)


def log_attn_entropy(step, ent_matrix, cfg):
    """Log attention entropy metrics to wandb."""
    ent = np.array(ent_matrix).mean(axis=0)  # avg devices → (depth, H)
    depth = cfg.depth
    alog = {}
    for li in [0, depth // 2, depth - 1]:
        alog[f'attn/entropy_layer{li}'] = float(ent[li].mean())
    for h in range(cfg.num_heads):
        alog[f'attn/entropy_head{h}_last'] = float(ent[-1, h])
    wandb.log(alog, step=step)
