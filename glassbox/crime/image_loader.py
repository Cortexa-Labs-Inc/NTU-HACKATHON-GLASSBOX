"""
UCF Crime PNG image loader.

Supports two common folder layouts:

  Layout A — ImageFolder (one sub-folder per class):
    root/
      Abuse/     *.png
      Arrest/    *.png
      Normal/    *.png
      ...

  Layout B — train/test pre-split:
    root/
      train/  Abuse/ ... Normal/ ...
      test/   Abuse/ ... Normal/ ...

Usage
-----
  loaders = load_ucf_crime_images('path/to/UCF_Crime', image_size=64)

HuggingFace reference dataset (64×64 PNG, 14 classes, ~1.3 M frames):
  https://huggingface.co/datasets/hibana2077/UCF-Crime-Dataset
"""

import os
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

# ── Canonical UCF Crime class names ─────────────────────────────────────────
UCF_CRIME_CLASSES = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
    'Explosion', 'Fighting', 'Normal', 'RoadAccidents',
    'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism',
]

# Alternate folder names that appear in some dataset versions
_ALIAS = {
    'Normal_Videos': 'Normal',
    'Normal Videos': 'Normal',
    'Road Accidents': 'RoadAccidents',
}


def get_transforms(image_size: int = 64, augment: bool = True):
    """
    Standard ImageNet normalisation.  Augmentation only for training.
    """
    normalise = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            normalise,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalise,
    ])


def _find_image_root(root: str) -> tuple[str, bool]:
    """
    Detect layout A vs B.  Returns (image_root, pre_split).
    pre_split=True  → root/train/ and root/test/ exist
    pre_split=False → root/ itself is the ImageFolder root
    """
    p = Path(root)
    has_train = (p / 'train').is_dir()
    has_test  = (p / 'test').is_dir()
    if has_train and has_test:
        return str(p), True
    return str(p), False


def load_ucf_crime_images(
    root: str,
    image_size: int = 64,
    batch_size: int = 32,
    val_frac: float  = 0.15,
    test_frac: float = 0.15,
    num_workers: int = 2,
    random_state: int = 42,
    max_samples_per_class: int = None,
) -> dict:
    """
    Load UCF Crime PNG frames from a folder root.

    Parameters
    ----------
    root                  : path to dataset root
    image_size            : resize target (64 for HuggingFace version)
    batch_size            : DataLoader batch size
    val_frac / test_frac  : fractions for val/test split (Layout A only)
    max_samples_per_class : cap per class for quick debugging (None = use all)

    Returns
    -------
    dict with keys:
        train_loader, val_loader, test_loader
        class_names  : list[str]
        n_classes    : int
        image_size   : int
        n_train, n_val, n_test : int
    """
    image_root, pre_split = _find_image_root(root)

    if pre_split:
        return _load_presplit(image_root, image_size, batch_size, num_workers,
                              max_samples_per_class)
    return _load_single_root(image_root, image_size, batch_size, val_frac,
                             test_frac, num_workers, random_state,
                             max_samples_per_class)


# ── Layout A: single root, we split ──────────────────────────────────────────

def _load_single_root(root, image_size, batch_size, val_frac, test_frac,
                      num_workers, random_state, max_per_class):
    train_tf = get_transforms(image_size, augment=True)
    eval_tf  = get_transforms(image_size, augment=False)

    full_ds = datasets.ImageFolder(root, transform=eval_tf)

    if max_per_class:
        full_ds = _cap_per_class(full_ds, max_per_class)

    n = len(full_ds)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    n_train = n - n_test - n_val

    generator = torch.Generator().manual_seed(random_state)
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test], generator=generator
    )

    # Apply augmentation only to train split
    train_ds.dataset = datasets.ImageFolder(root, transform=train_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    return {
        'train_loader': train_loader,
        'val_loader':   val_loader,
        'test_loader':  test_loader,
        'class_names':  full_ds.classes,
        'n_classes':    len(full_ds.classes),
        'image_size':   image_size,
        'n_train':      n_train,
        'n_val':        n_val,
        'n_test':       n_test,
    }


# ── Layout B: pre-split train/test ────────────────────────────────────────────

def _load_presplit(root, image_size, batch_size, num_workers, max_per_class):
    train_tf = get_transforms(image_size, augment=True)
    eval_tf  = get_transforms(image_size, augment=False)

    train_ds = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_tf)
    test_ds  = datasets.ImageFolder(os.path.join(root, 'test'),  transform=eval_tf)

    if max_per_class:
        train_ds = _cap_per_class(train_ds, max_per_class)
        test_ds  = _cap_per_class(test_ds,  max_per_class)

    # Carve val from train (last 15%)
    n_val   = int(len(train_ds) * 0.15)
    n_train = len(train_ds) - n_val
    train_ds, val_ds = random_split(train_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    return {
        'train_loader': train_loader,
        'val_loader':   val_loader,
        'test_loader':  test_loader,
        'class_names':  train_ds.dataset.classes,
        'n_classes':    len(train_ds.dataset.classes),
        'image_size':   image_size,
        'n_train':      n_train,
        'n_val':        n_val,
        'n_test':       len(test_ds),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cap_per_class(dataset: datasets.ImageFolder, max_per_class: int):
    """Return a Subset capped at max_per_class images per class."""
    from collections import defaultdict
    counts = defaultdict(int)
    indices = []
    for idx, (_, label) in enumerate(dataset.samples):
        if counts[label] < max_per_class:
            indices.append(idx)
            counts[label] += 1
    return Subset(dataset, indices)


def extract_features_from_loader(model, loader, device='cpu') -> tuple[np.ndarray, np.ndarray]:
    """
    Run CNN extractor on a DataLoader, return (features, labels) as numpy arrays.
    Used for collecting failure embeddings for the self-healing perturber.

    model : CrimeVisionGlassbox  (or just the extractor)
    Returns:
        X : (N, n_chunks * proj_dim)  — concatenated multi-scale features
        y : (N,)                       — class labels
    """
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats  = model.extract(images)   # (N, n_chunks * proj_dim)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.vstack(all_feats), np.concatenate(all_labels)
