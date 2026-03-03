"""
Online support encoder for FSDiT.

Takes support image paths from dataset batches, runs SigLIP2 online, and
maintains a small LRU cache to avoid recomputing repeated paths.
"""

import time
from collections import OrderedDict

import jax
import numpy as np
import tensorflow as tf

from encoder import SigLIP2Encoder


def _decode_path(path_val):
    if isinstance(path_val, bytes):
        return path_val.decode("utf-8")
    if isinstance(path_val, np.bytes_):
        return bytes(path_val).decode("utf-8")
    return str(path_val)


def _read_image(path, image_size):
    img = tf.io.read_file(path)
    try:
        img = tf.image.decode_jpeg(img, channels=3)
    except Exception:
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - 0.5) / 0.5
    return np.asarray(img.numpy(), dtype=np.float32)


class OnlineSupportEncoder:
    """SigLIP2 online encoder with LRU cache and batch dedup."""

    def __init__(
        self,
        variant="B/16",
        image_size=224,
        cache_items=1024,
        batch_size=256,
        no_pmap=False,
        warmup_need_seq=True,
        ckpt_path=None,
    ):
        self.image_size = int(image_size)
        self.cache_items = int(max(cache_items, 0))
        self.request_batch_size = int(max(batch_size, 1))
        self.cache = OrderedDict()
        self.siglip = SigLIP2Encoder.create(
            ckpt_path=ckpt_path,
            variant=variant,
            res=image_size,
        )

        self.n_dev = jax.local_device_count()
        self.use_pmap = (not no_pmap) and self.n_dev > 1
        if self.use_pmap:
            self.model_batch_size = int(
                ((self.request_batch_size + self.n_dev - 1) // self.n_dev) * self.n_dev
            )
        else:
            self.model_batch_size = self.request_batch_size

        if self.use_pmap:
            print(
                f"[Online SigLIP] pmap=on n_dev={self.n_dev} "
                f"request_bs={self.request_batch_size} model_bs={self.model_batch_size}"
            )
        else:
            print(f"[Online SigLIP] pmap=off model_bs={self.model_batch_size}")

        if self.use_pmap:
            self._encode_both_fn = jax.pmap(self.siglip._encode_both)
            self._encode_pooled_fn = jax.pmap(self.siglip._encode_batch)
        else:
            self._encode_both_fn = jax.jit(self.siglip._encode_both)
            self._encode_pooled_fn = jax.jit(self.siglip._encode_batch)

        # Warm up fixed-shape compiled paths once to avoid in-loop recompiles.
        warm = np.zeros(
            (self.model_batch_size, self.image_size, self.image_size, 3),
            dtype=np.float32,
        )
        _ = self._encode_chunk_fixed(warm, need_seq=bool(warmup_need_seq))

    def _cache_get(self, key):
        if self.cache_items <= 0:
            return None
        item = self.cache.get(key)
        if item is None:
            return None
        self.cache.move_to_end(key)
        return item

    def _cache_put(self, key, pooled, seq):
        if self.cache_items <= 0:
            return
        self.cache[key] = {
            "pooled": pooled.astype(np.float16, copy=False),
            "seq": None if seq is None else seq.astype(np.float16, copy=False),
        }
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_items:
            self.cache.popitem(last=False)

    def _encode_chunk_fixed(self, images_np, need_seq):
        if images_np.shape[0] != self.model_batch_size:
            raise ValueError(
                f"Expected fixed encode batch {self.model_batch_size}, got {images_np.shape[0]}"
            )
        if self.use_pmap:
            per_dev = self.model_batch_size // self.n_dev
            inp = images_np.reshape(self.n_dev, per_dev, self.image_size, self.image_size, 3)
            if need_seq:
                seq, pooled = self._encode_both_fn(inp)
                seq = np.asarray(jax.device_get(seq), dtype=np.float32).reshape(
                    self.model_batch_size, 196, 768
                )
                pooled = np.asarray(jax.device_get(pooled), dtype=np.float32).reshape(
                    self.model_batch_size, 768
                )
                return seq, pooled
            pooled = self._encode_pooled_fn(inp)
            pooled = np.asarray(jax.device_get(pooled), dtype=np.float32).reshape(
                self.model_batch_size, 768
            )
            return None, pooled

        if need_seq:
            seq, pooled = self._encode_both_fn(images_np)
            return (
                np.asarray(jax.device_get(seq), dtype=np.float32),
                np.asarray(jax.device_get(pooled), dtype=np.float32),
            )
        pooled = self._encode_pooled_fn(images_np)
        return None, np.asarray(jax.device_get(pooled), dtype=np.float32)

    def encode_paths(self, paths_2d, need_seq=True):
        """
        Args:
          paths_2d: (B, 5) np.ndarray/tensor-like of string/bytes paths
          need_seq: whether to return support sequence tokens
        Returns:
          supports_pooled: (B,5,768) float16
          supports_seq   : (B,5,196,768) float16, or None if need_seq=False
          stats dict for perf logging
        """
        t0 = time.time()
        paths_arr = np.asarray(paths_2d)
        if paths_arr.ndim != 2 or paths_arr.shape[1] != 5:
            raise ValueError(f"Expected support_paths shape (B,5), got {paths_arr.shape}")

        bsz = int(paths_arr.shape[0])
        flat_paths = [_decode_path(x) for x in paths_arr.reshape(-1)]
        ordered_unique = list(dict.fromkeys(flat_paths))

        hits = 0
        path_to_entry = {}
        misses = []
        for p in ordered_unique:
            entry = self._cache_get(p)
            if entry is not None and (not need_seq or entry["seq"] is not None):
                hits += 1
                path_to_entry[p] = entry
            else:
                misses.append(p)

        for st in range(0, len(misses), self.request_batch_size):
            chunk_paths = misses[st:st + self.request_batch_size]
            images_np = np.zeros(
                (self.model_batch_size, self.image_size, self.image_size, 3),
                dtype=np.float32,
            )
            for i, p in enumerate(chunk_paths):
                try:
                    images_np[i] = _read_image(p, self.image_size)
                except Exception as exc:
                    raise RuntimeError(f"Failed to decode support image: {p}") from exc
            seq_chunk, pooled_chunk = self._encode_chunk_fixed(images_np, need_seq=need_seq)
            for i, p in enumerate(chunk_paths):
                entry = {
                    "pooled": pooled_chunk[i].astype(np.float16, copy=False),
                    "seq": None if seq_chunk is None else seq_chunk[i].astype(np.float16, copy=False),
                }
                path_to_entry[p] = entry
                self._cache_put(p, entry["pooled"], entry["seq"])

        pooled_out = np.zeros((len(flat_paths), 768), dtype=np.float16)
        seq_out = np.zeros((len(flat_paths), 196, 768), dtype=np.float16) if need_seq else None

        for i, p in enumerate(flat_paths):
            entry = path_to_entry.get(p)
            if entry is None:
                raise RuntimeError(f"Support path not found after encode: {p}")
            pooled_out[i] = entry["pooled"]
            if need_seq:
                if entry["seq"] is None:
                    raise RuntimeError(f"Missing sequence embedding for path: {p}")
                seq_out[i] = entry["seq"]

        pooled_out = pooled_out.reshape(bsz, 5, 768)
        if need_seq:
            seq_out = seq_out.reshape(bsz, 5, 196, 768)
        unique_count = len(ordered_unique)
        stats = {
            "encode_time": float(time.time() - t0),
            "cache_hit_rate": float(hits / max(unique_count, 1)),
            "cache_items": int(len(self.cache)),
            "unique_paths_per_batch": int(unique_count),
            "miss_paths_per_batch": int(len(misses)),
        }
        return pooled_out, seq_out, stats
