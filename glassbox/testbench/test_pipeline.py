"""
CrimeVisionGlassbox — End-to-End Testbench
==========================================

Tests the full pipeline without a running API server:
  1. Load trained model from artefacts/
  2. Run inference on sample_frames/ (Normal + Anomaly)
  3. Print predictions, confidence, and chunk attribution scores
  4. Verify accuracy >= 80% on the 20 sample frames

Usage:
  cd glassbox/
  python3 testbench/test_pipeline.py
"""

import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from pathlib import Path
from PIL import Image

from crime.feature_extractor import CrimeVisionGlassbox
from crime.image_loader      import get_transforms

# ── Paths ──────────────────────────────────────────────────────────────────────
ARTEFACT_DIR   = Path(__file__).parent.parent / 'artefacts'
SAMPLE_DIR     = Path(__file__).parent / 'sample_frames'
META_PATH      = ARTEFACT_DIR / 'crime_meta.json'
MODEL_PATH     = ARTEFACT_DIR / 'crime_vision.pt'

# ── Load model ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  CrimeVisionGlassbox — Testbench")
print("=" * 60)

assert META_PATH.exists(),  f"Missing: {META_PATH}  (run training/crime_train.py first)"
assert MODEL_PATH.exists(), f"Missing: {MODEL_PATH} (run training/crime_train.py first)"

with open(META_PATH) as f:
    meta = json.load(f)

class_names = meta['class_names']
n_classes   = meta['n_classes']
chunk_names = meta['chunk_names']
image_size  = meta['image_size']
cfg         = meta['cfg']

print(f"\nModel info:")
print(f"  Classes:     {class_names}")
print(f"  Chunks:      {chunk_names}")
print(f"  Test AUC:    {meta['test_auc']}")
print(f"  Test Acc:    {meta['test_acc']}")
print(f"  Val AUC:     {meta['best_val_auc']}")

device = torch.device('cpu')
model = CrimeVisionGlassbox(
    n_classes=n_classes,
    proj_dim=cfg['proj_dim'],
    embed_dim=cfg['embed_dim'],
    backbone=cfg.get('backbone', 'tiny'),
    pretrained=False,
    freeze_backbone=cfg.get('freeze_backbone', False),
    use_ghost=cfg['use_ghost'],
    use_order_decomp=cfg['use_order_decomp'],
    n_sub_chunks=cfg.get('n_sub_chunks', 1),
)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
model.eval()
print(f"\nModel loaded from {MODEL_PATH.name}")

# ── Inference ──────────────────────────────────────────────────────────────────
transform = get_transforms(image_size, augment=False)

results = []
print(f"\n{'Frame':<35} {'True':>8} {'Pred':>8} {'Conf':>6}  Chunk blame")
print("-" * 80)

for true_label_name in ['Normal', 'Anomaly']:
    frame_dir = SAMPLE_DIR / true_label_name
    if not frame_dir.exists():
        print(f"  [skip] {frame_dir} not found")
        continue

    for img_path in sorted(frame_dir.glob('*.png')):
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].numpy()

        pred_idx  = int(np.argmax(probs))
        pred_name = class_names[pred_idx]
        confidence = float(probs[pred_idx])
        correct    = pred_name == true_label_name

        # Per-chunk blame using model's built-in attribution
        with torch.no_grad():
            feats = model.extract(tensor)   # (1, 128)
            attrib = model.glassbox.get_class_pair_contributions(
                feats, pred_idx, pred_idx   # blame toward predicted class
            )

        blame_str = "  ".join(
            f"{k}={v['pred_push']:+.2f}" for k, v in attrib.items()
        )

        mark = "✓" if correct else "✗"
        print(f"{mark} {img_path.name:<33} {true_label_name:>8} {pred_name:>8} "
              f"{confidence:5.1%}  {blame_str}")

        results.append({'correct': correct, 'true': true_label_name, 'pred': pred_name})

# ── Summary ────────────────────────────────────────────────────────────────────
n_correct = sum(r['correct'] for r in results)
n_total   = len(results)
accuracy  = n_correct / n_total if n_total > 0 else 0

# Baselines
n_normal  = sum(1 for r in results if r['true'] == 'Normal')
n_anomaly = sum(1 for r in results if r['true'] == 'Anomaly')
majority_class   = 'Normal' if n_normal >= n_anomaly else 'Anomaly'
majority_acc     = max(n_normal, n_anomaly) / n_total
random_acc       = (n_normal**2 + n_anomaly**2) / n_total**2  # P(agree) for random

print("\n" + "=" * 60)
print(f"  Accuracy on {n_total} sample frames: {n_correct}/{n_total} = {accuracy:.1%}")
print()
print(f"  Baselines (for context):")
print(f"    Majority-class ({majority_class} always): {majority_acc:.1%}")
print(f"    Random guessing (balanced):              {random_acc:.1%}")
print(f"    CrimeVisionGlassbox:                     {accuracy:.1%}   ← ours")
print(f"    Improvement over majority:               +{(accuracy - majority_acc)*100:.1f}pp")
print()

if accuracy >= 0.80:
    print("  PASS (>= 80% threshold)")
else:
    print("  WARN: accuracy below 80% — check model or sample frames")

print("=" * 60)
