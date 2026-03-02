"""
precompute_selfrecon_jax.py — k-shot precompute (JAX/Flax).

For each image, extract CLS + patch tokens via a vision encoder and build
ArrayRecord files for few-shot / self-reconstruction DiT training.

Encoder backends:
  - siglip2  (default): SigLIP2-B/16 via big_vision JAX checkpoint.
  - dinov2          : DINOv2-B/14 via HuggingFace transformers (PyTorch→numpy).

k-shot logic (--k K):
  k=0  — self-reconstruction: target conditions on its own CLS token.
  k>0  — few-shot: class images are tiled into non-overlapping groups of k+1.
         Each group produces k+1 records by rotating which image is the target;
         the remaining k images are supports whose CLS tokens are mean-pooled
         into a single condition vector.

Record schema (msgpack):
  target_path     : str
  class_id        : int
  supports_pooled : bytes  (1, D) float16  — mean-pooled CLS of supports
  supports_seq    : bytes  (k, P, D) float16  — stacked patch tokens of supports
                           empty bytes when --no-keep_patches is set

Designed for Kaggle TPU v5e-8.

Usage:
    # Self-recon, SigLIP2
    python precompute_selfrecon_jax.py \
        --data_dir /kaggle/working/miniimagenet_split \
        --out_dir /kaggle/working/precomp_k0 \
        --splits train,val --batch_size 64

    # 5-shot, DINOv2, no patch tokens
    python precompute_selfrecon_jax.py \
        --data_dir /kaggle/working/miniimagenet_split \
        --out_dir /kaggle/working/precomp_k5_dinov2 \
        --splits train,val --batch_size 64 \
        --encoder dinov2 --k 5 --no-keep_patches
"""

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import msgpack
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from array_record.python.array_record_module import ArrayRecordWriter
except ImportError:
    raise ImportError("Install array_record: pip install array_record")


# ═══════════════════════════════════════════════════════════════════════════════
#  SigLIP2 JAX encoder (via big_vision)
# ═══════════════════════════════════════════════════════════════════════════════

def setup_big_vision():
    repo = os.environ.get('BIG_VISION_DIR', '/kaggle/working/big_vision')
    if not os.path.exists(repo):
        print(f"Cloning big_vision → {repo}")
        os.system(f'git clone --quiet --branch=main --depth=1 '
                  f'https://github.com/google-research/big_vision {repo} > /dev/null 2>&1')
        os.system(f'pip install -q -r {repo}/big_vision/requirements.txt > /dev/null 2>&1')
    if repo not in sys.path:
        sys.path.insert(0, repo)


def create_siglip2_jax(variant='B/16', res=224):
    """Create SigLIP2 JAX model and load weights."""
    setup_big_vision()
    import big_vision.models.proj.image_text.two_towers as m
    import ml_collections

    txt_var, patch = variant.split('/')
    emb_dim = {'B': 768, 'L': 1024, 'So400m': 1152}[txt_var]

    cfg = ml_collections.ConfigDict(dict(
        image_model='vit',
        image=dict(pool_type='map', scan=True, variant=variant),
        text_model='proj.image_text.text_transformer',
        text=dict(scan=True, variant=txt_var, vocab_size=256_000),
        out_dim=[None, emb_dim],
        bias_init=-10,
    ))

    name = f'siglip2_{txt_var.lower()}{patch}_{res}.npz'
    ckpt_path = f'/tmp/{name}'
    if not os.path.exists(ckpt_path):
        url = f'https://storage.googleapis.com/big_vision/siglip2/{name}'
        print(f"Downloading {url}")
        os.system(f'wget -q {url} -O {ckpt_path}')

    print(f"Loading SigLIP2 {variant} {res}×{res}")
    model = m.Model(**cfg)
    params = m.load(None, ckpt_path, cfg)
    return model, params, emb_dim


def make_encode_fn(model, params, emb_dim):
    """Create JIT-compiled encode function returning (seq, pooled)."""

    @jax.jit
    def encode_batch(images):
        """(N, 224, 224, 3) → (seq: (N, 196, emb), pooled: (N, emb))"""
        outputs = model.apply({'params': params}, images, None)

        # Parse outputs to find sequence and pooled tensors
        leaves = jax.tree.leaves(outputs)
        bsz = images.shape[0]
        pooled = None
        seq = None
        for arr in leaves:
            shape = arr.shape
            if len(shape) == 2 and shape[0] == bsz and shape[1] == emb_dim:
                if pooled is None:
                    pooled = arr
            if len(shape) == 3 and shape[0] == bsz and shape[-1] == emb_dim and shape[1] == 196:
                if seq is None:
                    seq = arr
        return jax.lax.stop_gradient(seq), jax.lax.stop_gradient(pooled)

    return encode_batch


# ═══════════════════════════════════════════════════════════════════════════════
#  DINOv2-B encoder (HuggingFace transformers)
# ═══════════════════════════════════════════════════════════════════════════════

def create_dinov2_encoder(model_name='facebook/dinov2-base'):
    """
    Load DINOv2-B from HuggingFace and return a numpy-based encode function.

    Returns:
        encode_fn : callable(images: np.ndarray (N,H,W,3) float32 [0,1])
                    → (seq: np.ndarray (N, num_patches, 768),
                       pooled: np.ndarray (N, 768))
        emb_dim   : int  (768 for DINOv2-B)
        num_patches: int (256 for 224×224 / patch14)
    """
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError:
        raise ImportError(
            "Install transformers & torch: pip install transformers torch"
        )

    print(f"Loading DINOv2 '{model_name}' from HuggingFace…")
    processor = AutoImageProcessor.from_pretrained(model_name)
    hf_model = AutoModel.from_pretrained(model_name)
    hf_model.eval()

    # Use GPU if available, else CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hf_model = hf_model.to(device)
    print(f"  DINOv2 running on: {device}")

    emb_dim = hf_model.config.hidden_size          # 768
    patch_size = hf_model.config.patch_size         # 14
    # num_patches depends on image_size set later; we'll compute dynamically

    def encode_fn_dinov2(images: np.ndarray):
        """
        images: (N, H, W, 3) float32 in [0, 1].
        Returns (seq, pooled) as np.ndarray float32.
        """
        # Convert [0,1] → uint8 PIL images expected by processor
        imgs_pil = [
            Image.fromarray((img * 255).astype(np.uint8))
            for img in images
        ]
        inputs = processor(images=imgs_pil, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = hf_model(**inputs)

        # last_hidden_state: (N, 1 + num_patches, emb_dim)
        hidden = out.last_hidden_state.cpu().numpy()   # (N, 1+P, D)
        pooled = hidden[:, 0, :]                        # CLS  (N, D)
        seq = hidden[:, 1:, :]                          # patches (N, P, D)
        return seq, pooled

    return encode_fn_dinov2, emb_dim, patch_size


# ═══════════════════════════════════════════════════════════════════════════════
#  Image loading
# ═══════════════════════════════════════════════════════════════════════════════

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPEG'}


def collect_class_images(split_dir):
    """Returns dict: class_name → sorted list of image paths."""
    classes = {}
    for cls_name in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        imgs = sorted([
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if os.path.splitext(f)[1] in IMAGE_EXTS
        ])
        if imgs:
            classes[cls_name] = imgs
    return classes


def load_and_preprocess(path, image_size=224):
    """Load image as normalized float32 array for SigLIP2."""
    img = Image.open(path).convert('RGB')
    img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr  # (224, 224, 3), [0, 1]


def load_and_preprocess_target(path, image_size=224):
    """Load image as [-1, 1] for DiT training target."""
    img = Image.open(path).convert('RGB')
    img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # → [-1, 1]
    return arr


# ═══════════════════════════════════════════════════════════════════════════════
#  Record building — k-shot + self-recon
# ═══════════════════════════════════════════════════════════════════════════════

def _encode_all(paths, encode_fn, batch_size, image_size, keep_patches):
    """
    Encode a list of image paths in batches.
    Returns:
        cls_tokens   : np.ndarray (N, D)      float32
        patch_tokens : np.ndarray (N, P, D)   float32  or None
    """
    all_cls, all_patches = [], []
    for i in range(0, len(paths), batch_size):
        batch = np.stack([
            load_and_preprocess(p, image_size) for p in paths[i:i + batch_size]
        ])
        seq, pooled = encode_fn(batch)
        all_cls.append(np.array(pooled))
        if keep_patches:
            all_patches.append(np.array(seq))
    cls_tokens = np.concatenate(all_cls, axis=0)          # (N, D)
    patch_tokens = np.concatenate(all_patches, axis=0) if keep_patches else None
    return cls_tokens, patch_tokens


def _make_record(target_path, class_id, support_cls, support_patches):
    """
    Pack one record.

    Args:
        support_cls     : (k, D) float32  — CLS tokens of k support images
                          (k=1 for self-recon)
        support_patches : (k, P, D) float32  or None
    Returns bytes (msgpack).
    """
    pooled_cond = support_cls.mean(axis=0, keepdims=True).astype(np.float16)  # (1, D)
    if support_patches is not None:
        # (k, P, D)  — stacked patch sequences of all supports
        seq_cond = support_patches.astype(np.float16)
    else:
        seq_cond = np.zeros((0,), dtype=np.float16)  # empty placeholder

    record = {
        "target_path": target_path,
        "class_id": int(class_id),
        "supports_pooled": pooled_cond.tobytes(),
        "supports_seq": seq_cond.tobytes(),
    }
    return msgpack.packb(record, use_bin_type=True)


def build_split(split, data_dir, out_dir, encode_fn, batch_size, image_size,
               k=0, keep_patches=True):
    """
    Build ArrayRecord for one split.

    k=0  — self-reconstruction: image conditions on itself.
    k>0  — few-shot: class images → non-overlapping groups of k+1;
            each group produces k+1 records (rotating target);
            condition = mean-pooled CLS of the k supports.
    """
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        print(f"[Skip] split '{split}' not found at {split_dir}")
        return

    class_images = collect_class_images(split_dir)
    class_names = sorted(class_images.keys())
    class_to_id = {name: i for i, name in enumerate(class_names)}

    set_size = k + 1  # images per group
    mode = "self-recon" if k == 0 else f"{k}-shot"
    print(f"[{split}] {len(class_names)} classes | mode={mode} | keep_patches={keep_patches}")

    os.makedirs(out_dir, exist_ok=True)
    arecord_path = os.path.join(out_dir, f"{split}.arecord")
    writer = ArrayRecordWriter(arecord_path, "group_size:1")
    total_records = 0

    for cls_name in tqdm(class_names, desc=f"{split}-classes"):
        class_id = class_to_id[cls_name]
        imgs = class_images[cls_name]              # sorted list of paths

        if k == 0:
            # ── self-reconstruction ──────────────────────────────────────────
            cls_tokens, patch_tokens = _encode_all(
                imgs, encode_fn, batch_size, image_size, keep_patches)
            print(f"  [{cls_name}] self-recon: {len(imgs)} images encoded")

            for idx, path in enumerate(imgs):
                sup_cls = cls_tokens[idx:idx + 1]            # (1, D)
                sup_pat = patch_tokens[idx:idx + 1] if keep_patches else None
                writer.write(_make_record(path, class_id, sup_cls, sup_pat))
                total_records += 1

        else:
            # ── k-shot ───────────────────────────────────────────────────────
            num_sets = len(imgs) // set_size
            if num_sets == 0:
                print(f"  [{cls_name}] WARNING: only {len(imgs)} images, "
                      f"need {set_size} for k={k}. Skipping.")
                continue
            imgs = imgs[:num_sets * set_size]    # drop remainder

            cls_tokens, patch_tokens = _encode_all(
                imgs, encode_fn, batch_size, image_size, keep_patches)
            print(f"  [{cls_name}] {num_sets} sets × {set_size} → {num_sets * set_size} records")

            for s in range(num_sets):
                set_idx = list(range(s * set_size, (s + 1) * set_size))
                for target_pos in range(set_size):
                    target_idx = set_idx[target_pos]
                    support_idx = [set_idx[j] for j in range(set_size) if j != target_pos]

                    sup_cls = cls_tokens[support_idx]           # (k, D)
                    sup_pat = patch_tokens[support_idx] if keep_patches else None  # (k,P,D)
                    writer.write(_make_record(
                        imgs[target_idx], class_id, sup_cls, sup_pat))
                    total_records += 1

    writer.close()

    # Metadata
    meta = {
        "split": split,
        "mode": mode,
        "k": k,
        "keep_patches": keep_patches,
        "num_classes": len(class_names),
        "total_records": total_records,
        "format": "arrayrecord",
        "serialization": "msgpack",
        "class_names": class_names,
    }
    with open(os.path.join(out_dir, f"{split}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    file_mb = os.path.getsize(arecord_path) / (1024 * 1024)
    print(f"[{split}] Written {total_records} records → {arecord_path} ({file_mb:.1f} MB)")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="k-shot precompute: build ArrayRecord from a vision encoder."
    )
    parser.add_argument('--data_dir', required=True,
                        help='Split root with train/val/test sub-dirs.')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for ArrayRecord files.')
    parser.add_argument('--splits', default='train,val',
                        help='Comma-separated splits to process.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Encoding batch size.')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Resize images to this resolution.')
    parser.add_argument('--variant', default='B/16',
                        help='SigLIP2 variant (ignored for DINOv2).')
    parser.add_argument(
        '--encoder', default='siglip2', choices=['siglip2', 'dinov2'],
        help='Vision encoder: siglip2 (JAX, default) or dinov2 (HuggingFace).',
    )
    parser.add_argument(
        '--k', type=int, default=0,
        help=(
            'Number of support images per record. '
            'k=0: self-reconstruction (image conditions on itself). '
            'k>0: few-shot; each class is split into non-overlapping groups '
            'of k+1 images, producing k+1 records per group by rotating the target.'
        ),
    )
    parser.add_argument(
        '--keep_patches', default=True,
        action=argparse.BooleanOptionalAction,
        help='Store patch token sequences in records (default: True). '
             'Use --no-keep_patches to save space when only CLS is needed.',
    )
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"Num devices: {jax.device_count()}")
    print(f"Encoder     : {args.encoder}")
    print(f"k           : {args.k}")
    print(f"keep_patches: {args.keep_patches}")

    # ── Build encoder ────────────────────────────────────────────────────────
    if args.encoder == 'siglip2':
        model, params, emb_dim = create_siglip2_jax(args.variant, args.image_size)
        encode_fn = make_encode_fn(model, params, emb_dim)
        print("Warmup JIT (SigLIP2)…")
        dummy = np.zeros((1, args.image_size, args.image_size, 3), dtype=np.float32)
        _ = encode_fn(dummy)
        print("Warmup done.")
    elif args.encoder == 'dinov2':
        encode_fn, emb_dim, _patch_size = create_dinov2_encoder()
        print("Warmup DINOv2…")
        dummy = np.zeros((1, args.image_size, args.image_size, 3), dtype=np.float32)
        _seq, _cls = encode_fn(dummy)
        print(f"Warmup done. patches={_seq.shape[1]}, emb_dim={emb_dim}")
    else:
        raise ValueError(f"Unknown encoder: {args.encoder}")

    # ── Process splits ───────────────────────────────────────────────────────
    for split in args.splits.split(','):
        split = split.strip()
        if split:
            t0 = time.time()
            build_split(
                split, args.data_dir, args.out_dir,
                encode_fn, args.batch_size, args.image_size,
                k=args.k, keep_patches=args.keep_patches,
            )
            print(f"  {split} took {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
