"""
CrimeVisionGlassbox training script.

Trains CrimeVisionGlassbox (ResNet18 feature extractor + GlassboxNetV2)
on UCF Crime PNG frames, then runs self-healing rounds.

Workflow
--------
  1. Load PNG frames via torchvision.ImageFolder
  2. Extract multi-scale CNN features per epoch (frozen ResNet18 backbone)
  3. Train Glassbox on features: ChunkNets + Ghost gates + order decomp
  4. Run self-healing rounds: cluster val failures → Gaussian perturb → retrain
  5. Save all artefacts

Usage
-----
  python3 training/crime_train.py

Set dataset path and hyperparameters in CFG below.

Dataset
-------
  UCF Crime PNG frames — one sub-folder per class:
    data_root/
      Abuse/     *.png
      Normal/    *.png
      ...
  HuggingFace: https://huggingface.co/datasets/hibana2077/UCF-Crime-Dataset
  Kaggle:      https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset

Artefacts saved to artefacts/:
  crime_vision.pt              — full model weights (extractor + Glassbox)
  crime_meta.json              — chunk config, class names, metrics
  crime_train_features.npz     — cached CNN features for API / self-heal
  crime_training_history.json  — epoch curves + healing records
"""

import os
import sys
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crime.image_loader     import load_ucf_crime_images, extract_features_from_loader
from crime.feature_extractor import CrimeVisionGlassbox
from crime.self_heal         import SelfHealingLoop

ARTEFACT_DIR = os.path.join(os.path.dirname(__file__), '..', 'artefacts')
os.makedirs(ARTEFACT_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
CFG = {
    # Data
    'data_root':   'CUHK_Avenue',  # CUHK Avenue binary: Normal / Anomaly
    'image_size':  64,
    'batch_size':  32,
    'num_workers': 2,
    # Optionally cap frames per class (set None to use all):
    'max_per_class': None,         # ← None = use all available (300/class for synthetic)

    # CNN extractor  ('tiny' = fast CPU, no download; 'resnet18' = full backbone)
    'backbone':        'tiny',
    'proj_dim':        32,         # feature dimension per chunk (4 chunks → 128 total)
    'pretrained':      False,
    'freeze_backbone': False,

    # Glassbox
    'embed_dim':        16,
    'use_ghost':        True,
    'use_order_decomp': True,
    'n_sub_chunks':     3,     # MoE sub-experts per named chunk (1 = off)

    # Training  (tuned for 300-image/class synthetic subset; scale up for full dataset)
    'epochs':       40,
    'lr':           1e-3,
    'lambda_peak':  0.006,
    'warmup_frac':  0.20,
    'hold_frac':    0.60,

    # Self-healing
    'self_heal':       True,
    'heal_rounds':     3,
    'heal_n_clusters': 5,
    'heal_n_synth':    40,
    'heal_sigma':      0.3,
    'heal_epochs':     15,
    'heal_patience':   2,
}


def lambda_schedule(epoch, n_epochs, peak, warmup=0.2, hold=0.6):
    frac = epoch / n_epochs
    if frac < warmup:
        return peak * (frac / warmup)
    if frac < hold:
        return peak
    return peak * 0.5 * (1.0 + np.cos(np.pi * (frac - hold) / (1.0 - hold)))


def eval_loader(model, loader, device) -> tuple[float, float]:
    model.eval()
    all_probs, all_preds, all_true = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)
            all_true.append(labels.numpy())
    all_probs = np.vstack(all_probs)
    y_true    = np.concatenate(all_true)
    y_pred    = np.concatenate(all_preds)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, all_probs, multi_class='ovr')
    except ValueError:
        auc = float(acc)
    return round(float(auc), 4), round(float(acc), 4)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 64)
    print("  CrimeVisionGlassbox Training")
    print(f"  Device: {device}")
    print("=" * 64)

    # ── Load images ────────────────────────────────────────────────────────────
    root = os.path.join(os.path.dirname(__file__), '..', CFG['data_root'])
    data = load_ucf_crime_images(
        root=root,
        image_size=CFG['image_size'],
        batch_size=CFG['batch_size'],
        num_workers=CFG['num_workers'],
        max_samples_per_class=CFG['max_per_class'],
    )
    class_names = data['class_names']
    n_classes   = data['n_classes']
    print(f"\nClasses ({n_classes}): {class_names}")
    print(f"Split: train={data['n_train']}, val={data['n_val']}, test={data['n_test']}")

    # ── Build model ────────────────────────────────────────────────────────────
    model = CrimeVisionGlassbox(
        n_classes=n_classes,
        proj_dim=CFG['proj_dim'],
        embed_dim=CFG['embed_dim'],
        backbone=CFG.get('backbone', 'tiny'),
        pretrained=CFG['pretrained'],
        freeze_backbone=CFG['freeze_backbone'],
        use_ghost=CFG['use_ghost'],
        use_order_decomp=CFG['use_order_decomp'],
        n_sub_chunks=CFG.get('n_sub_chunks', 1),
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Model: {total:,} total params, {trainable:,} trainable "
          f"({'backbone frozen' if CFG['freeze_backbone'] else 'full fine-tune'})")

    # Frozen backbone → train proj layers + full Glassbox
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=CFG['lr']
    )

    # Weighted loss to handle class imbalance (e.g. 10:1 Normal:Anomaly)
    from collections import Counter
    label_counts = Counter(label for _, label in data['train_loader'].dataset)
    total_samples = sum(label_counts.values())
    class_weights = torch.tensor(
        [total_samples / (n_classes * label_counts.get(i, 1)) for i in range(n_classes)],
        dtype=torch.float, device=device
    )
    print(f"Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_auc = 0.0
    best_state   = None
    train_history = []

    # ── Training loop ──────────────────────────────────────────────────────────
    print("\nTraining ...")
    for epoch in range(1, CFG['epochs'] + 1):
        lam = lambda_schedule(epoch, CFG['epochs'], CFG['lambda_peak'],
                              CFG['warmup_frac'], CFG['hold_frac'])
        model.train()
        train_loss = 0.0
        for imgs, labels in data['train_loader']:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            if CFG['use_ghost']:
                loss = loss + lam * model.get_gate_l1_loss()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_auc, val_acc = eval_loader(model, data['val_loader'], device)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = copy.deepcopy(model.state_dict())

        rec = {
            'epoch': epoch, 'val_auc': val_auc, 'val_acc': val_acc,
            'lambda': round(lam, 5),
            'train_loss': round(train_loss / len(data['train_loader']), 4),
        }
        train_history.append(rec)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | val_auc={val_auc:.4f} | "
                  f"val_acc={val_acc:.4f} | λ={lam:.5f}")

    model.load_state_dict(best_state)
    print(f"\nBest val AUC: {best_val_auc:.4f}")

    # ── Test evaluation ────────────────────────────────────────────────────────
    test_auc, test_acc = eval_loader(model, data['test_loader'], device)
    print(f"Test AUC: {test_auc:.4f}  |  Test Acc: {test_acc:.4f}")

    # ── Cache CNN feature vectors for self-healing (no GPU needed at heal time) ─
    print("\nCaching CNN features for self-healing ...")
    model.eval()
    model = model.cpu()

    X_train_feats, y_train = extract_features_from_loader(model, data['train_loader'])
    X_val_feats,   y_val   = extract_features_from_loader(model, data['val_loader'])
    X_test_feats,  y_test  = extract_features_from_loader(model, data['test_loader'])
    print(f"  Cached: train={X_train_feats.shape}, val={X_val_feats.shape}, "
          f"test={X_test_feats.shape}")

    # ── Self-healing rounds ────────────────────────────────────────────────────
    heal_history = []
    if CFG['self_heal']:
        print("\nStarting self-healing rounds ...")

        # Self-healing operates on cached feature vectors (not images)
        # The model here is the CrimeGlassboxNet (Glassbox-only, no CNN extractor)
        # because the extractor is frozen and features are pre-computed
        glassbox_only = model.glassbox

        healer = SelfHealingLoop(
            model=glassbox_only,
            X_train=X_train_feats,
            y_train=y_train,
            X_val=X_val_feats,
            y_val=y_val,
            class_names=class_names,
            chunk_names=model.chunk_names,
            n_clusters=CFG['heal_n_clusters'],
            n_synthetic=CFG['heal_n_synth'],
            sigma_scale=CFG['heal_sigma'],
            epochs_per_round=CFG['heal_epochs'],
            max_rounds=CFG['heal_rounds'],
            patience=CFG['heal_patience'],
        )
        heal_history = healer.run()

        # Restore healed Glassbox weights back into the vision model
        model.glassbox.load_state_dict(healer.model.state_dict())

    # ── Save artefacts ─────────────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(ARTEFACT_DIR, 'crime_vision.pt'))

    np.savez(
        os.path.join(ARTEFACT_DIR, 'crime_train_features.npz'),
        X_train=X_train_feats, X_val=X_val_feats, X_test=X_test_feats,
        y_train=y_train,       y_val=y_val,        y_test=y_test,
    )

    meta = {
        'chunk_names':    model.chunk_names,
        'chunk_sizes':    [CFG['proj_dim']] * 4,
        'n_classes':      n_classes,
        'class_names':    class_names,
        'n_features':     CFG['proj_dim'] * 4,
        'proj_dim':       CFG['proj_dim'],
        'image_size':     CFG['image_size'],
        'cfg':            CFG,
        'test_auc':       test_auc,
        'test_acc':       test_acc,
        'best_val_auc':   round(best_val_auc, 4),
        'n_train':        data['n_train'],
        'n_val':          data['n_val'],
        'n_test':         data['n_test'],
    }
    with open(os.path.join(ARTEFACT_DIR, 'crime_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(ARTEFACT_DIR, 'crime_training_history.json'), 'w') as f:
        json.dump({'training': train_history, 'healing': heal_history}, f, indent=2)

    print(f"\nArtefacts saved to {ARTEFACT_DIR}/")
    print("  crime_vision.pt, crime_meta.json")
    print("  crime_train_features.npz, crime_training_history.json")

    return model, meta


if __name__ == '__main__':
    train()
