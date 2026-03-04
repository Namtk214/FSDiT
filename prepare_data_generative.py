"""
prepare_data_generative.py — Split miniImageNet for generative modeling.

For generative models, we want ALL classes in train/val/test,
but split SAMPLES within each class (not disjoint classes).

This ensures:
- Model trains on all classes
- FID evaluation measures generation quality on seen classes
- Val/test splits are for monitoring overfitting, not few-shot learning

Usage:
    python prepare_data_generative.py \
        --src /kaggle/input/datasets/arjunashok33/miniimagenet \
        --dst ./data/miniimagenet_gen \
        --train_ratio 0.8 \
        --val_ratio 0.1 \
        --test_ratio 0.1 \
        --seed 42
"""

import os
import shutil
import argparse
import numpy as np
from pathlib import Path


def split_class_samples(class_dir, train_ratio, val_ratio, test_ratio, seed):
    """
    Split samples within a single class directory.

    Returns:
        dict: {'train': [file1, file2, ...], 'val': [...], 'test': [...]}
    """
    # Get all image files
    all_files = sorted([
        f for f in os.listdir(class_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG'))
    ])

    n_total = len(all_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # Rest goes to test

    # Shuffle with seed
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)

    splits = {
        'train': [all_files[i] for i in indices[:n_train]],
        'val': [all_files[i] for i in indices[n_train:n_train + n_val]],
        'test': [all_files[i] for i in indices[n_train + n_val:]],
    }

    return splits


def main():
    parser = argparse.ArgumentParser(
        description='Split miniImageNet samples (not classes) for generative modeling.'
    )
    parser.add_argument('--src', required=True, help='Source directory with class folders.')
    parser.add_argument('--dst', required=True, help='Destination directory for split.')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train split ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Val split ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test split ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--copy', action='store_true', help='Copy files instead of symlinking.')
    parser.add_argument('--max_classes', type=int, default=None, help='Limit number of classes (for testing)')
    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    assert abs(total_ratio - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {total_ratio}"

    # List all class directories
    all_classes = sorted([
        d for d in os.listdir(args.src)
        if os.path.isdir(os.path.join(args.src, d))
    ])

    if args.max_classes:
        all_classes = all_classes[:args.max_classes]

    print(f"Found {len(all_classes)} classes in {args.src}")
    print(f"Split ratios: train={args.train_ratio:.1%}, val={args.val_ratio:.1%}, test={args.test_ratio:.1%}")
    print(f"Seed: {args.seed}")
    print()

    # Create split directories
    for split_name in ['train', 'val', 'test']:
        os.makedirs(os.path.join(args.dst, split_name), exist_ok=True)

    total_samples = {'train': 0, 'val': 0, 'test': 0}

    # Process each class
    for cls_idx, cls_name in enumerate(all_classes):
        src_class_dir = os.path.join(args.src, cls_name)

        # Split samples within this class
        splits = split_class_samples(
            src_class_dir,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            seed=args.seed + cls_idx  # Different seed per class
        )

        # Create class directories in each split and link/copy files
        for split_name, file_list in splits.items():
            dst_class_dir = os.path.join(args.dst, split_name, cls_name)
            os.makedirs(dst_class_dir, exist_ok=True)

            for filename in file_list:
                src_file = os.path.join(src_class_dir, filename)
                dst_file = os.path.join(dst_class_dir, filename)

                if os.path.exists(dst_file):
                    continue

                if args.copy:
                    shutil.copy2(src_file, dst_file)
                else:
                    os.symlink(os.path.abspath(src_file), dst_file)

            total_samples[split_name] += len(file_list)

        if (cls_idx + 1) % 10 == 0 or (cls_idx + 1) == len(all_classes):
            print(f"  Processed {cls_idx + 1}/{len(all_classes)} classes...")

    print()
    print("=" * 60)
    print("Split Summary:")
    print("=" * 60)
    print(f"Classes: {len(all_classes)} (ALL classes in each split)")
    for split_name in ['train', 'val', 'test']:
        n_samples = total_samples[split_name]
        split_dir = os.path.join(args.dst, split_name)
        print(f"  {split_name:5s}: {n_samples:5d} samples → {split_dir}")

    print()
    print(f"✓ Done! Use --data_dir {args.dst} when training")
    print()
    print("Note: This is SAMPLE-level split (all classes in each split).")
    print("      Different from few-shot learning class-disjoint split.")


if __name__ == '__main__':
    main()
