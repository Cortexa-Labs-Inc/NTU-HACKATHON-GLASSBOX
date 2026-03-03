#!/usr/bin/env python3
"""
CrimeVisionGlassbox — Ablation Study
=====================================
Baseline and component ablations using cached CNN features.

Usage:
    cd glassbox/
    python3 ablation/run_ablation.py

All variants train on the same cached CNN features (artefacts/crime_train_features.npz)
so the CNN backbone is held fixed — only the Glassbox head changes per variant.
Estimated runtime: ~15 min on CPU.
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.glassbox_net_v2 import GlassboxNetV2
from crime.self_heal import SelfHealingLoop

# ── Configuration ─────────────────────────────────────────────────────────────
CHUNK_SIZES = [32, 32, 32, 32]   # proj_dim=32, 4 chunks → 128D total
EMBED_DIM   = 16
N_CLASSES   = 2
EPOCHS      = 25                  # Reduced from 40 — same for all variants (fair)
LR          = 1e-3
BATCH       = 32
HEAL_ROUNDS = 2                   # Reduced from 3 — enough to show the effect
HEAL_EPOCHS = 10
SEED        = 42
LAMBDA_GATE = 0.006               # Ghost gate L1 regularisation (same as full training)

CLASS_NAMES = ['Anomaly', 'Normal']
CHUNK_NAMES = ['Texture', 'Structure', 'Context', 'Semantic']

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_features():
    path = os.path.join(ROOT, 'artefacts', 'crime_train_features.npz')
    d = np.load(path)
    return (
        d['X_train'], d['y_train'],
        d['X_val'],   d['y_val'],
        d['X_test'],  d['y_test'],
    )


def get_class_weights(y: np.ndarray) -> torch.Tensor:
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return torch.FloatTensor(weights)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model, X_train, y_train, epochs=EPOCHS, lr=LR,
                class_weights=None, lambda_gate=LAMBDA_GATE):
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True)
    crit    = nn.CrossEntropyLoss(weight=class_weights)
    opt     = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        for X_b, y_b in loader:
            opt.zero_grad()
            logits = model(X_b)
            loss   = crit(logits, y_b)
            if hasattr(model, 'get_gate_l1_loss'):
                loss = loss + lambda_gate * model.get_gate_l1_loss()
            loss.backward()
            opt.step()
    return model


def eval_auc(model, X, y) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X))
        probs  = torch.softmax(logits, dim=1).numpy()
    # probs[:,1] = P(Normal); higher for Normal frames → AUC consistent with training code
    return roc_auc_score(y, probs[:, 1])


# ── Baseline model factories ──────────────────────────────────────────────────

def build_linear():
    """Single linear layer — maximum simplicity."""
    return nn.Linear(128, N_CLASSES)


def build_mlp():
    """Standard 2-hidden-layer MLP — no chunk decomposition."""
    return nn.Sequential(
        nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
        nn.Linear(128, N_CLASSES),
    )


def build_glassbox(use_ghost=True, use_order_decomp=True, n_sub_chunks=3):
    return GlassboxNetV2(
        chunk_sizes=CHUNK_SIZES,
        embed_dim=EMBED_DIM,
        n_classes=N_CLASSES,
        use_ghost=use_ghost,
        use_order_decomp=use_order_decomp,
        n_sub_chunks=n_sub_chunks,
    )


# ── Run one variant ───────────────────────────────────────────────────────────

def run_variant(name, model, X_train, y_train, X_val, y_val, X_test, y_test,
                class_weights, heal=False) -> dict:
    t0 = time.time()
    print(f"\n  [{name}]")
    sys.stdout.flush()

    model = train_model(model, X_train, y_train, class_weights=class_weights)

    if heal:
        healer = SelfHealingLoop(
            model=model,
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            class_names=CLASS_NAMES,
            chunk_names=CHUNK_NAMES,
            n_clusters=5,
            n_synthetic=40,
            sigma_scale=0.3,
            lr=LR,
            epochs_per_round=HEAL_EPOCHS,
            max_rounds=HEAL_ROUNDS,
            patience=2,
            lambda_gate=0.004,
        )
        healer.run()

    val_auc  = eval_auc(model, X_val,  y_val)
    test_auc = eval_auc(model, X_test, y_test)
    elapsed  = time.time() - t0

    print(f"    Val AUC: {val_auc:.4f}  |  Test AUC: {test_auc:.4f}  |  {elapsed:.0f}s")
    return {
        'name':      name,
        'val_auc':   round(val_auc,  4),
        'test_auc':  round(test_auc, 4),
        'elapsed_s': round(elapsed,  1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_features()
    cw = get_class_weights(y_train)

    print("=" * 60)
    print("   CRIMEVISIONGLASSBOX — ABLATION STUDY")
    print("=" * 60)
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}D (4 chunks × 32)  fixed CNN backbone")
    print(f"  Epochs: {EPOCHS}  Heal rounds: {HEAL_ROUNDS}  Seed: {SEED}")

    results = []

    # ── BASELINE ABLATION ──────────────────────────────────────────────────────
    print("\n\n── BASELINE ABLATION ──────────────────────────────────────────")
    print("  (Same CNN features; simpler classifiers on top)\n")

    for label, factory, heal in [
        ('Linear Baseline',            build_linear,  False),
        ('MLP Baseline',               build_mlp,     False),
        ('GlassboxNet (no healing)',   lambda: build_glassbox(), False),
        ('Full GlassboxNet + Healing', lambda: build_glassbox(), True),
    ]:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        r = run_variant(label, factory(), X_train, y_train,
                        X_val, y_val, X_test, y_test, cw, heal=heal)
        results.append(r)

    # ── COMPONENT ABLATION ────────────────────────────────────────────────────
    print("\n\n── COMPONENT ABLATION ─────────────────────────────────────────")
    print("  (Full GlassboxNet with one component removed at a time)\n")

    component_variants = [
        # (name, ghost, order_decomp, n_sub_chunks, heal)
        ('Full System (reference)',     True,  True,  3, True),
        ('w/o Ghost Gates',             False, True,  3, True),
        ('w/o Order Decomposition',     True,  False, 3, True),
        ('w/o MoSE  (K=1)',             True,  True,  1, True),
        ('w/o Self-Healing',            True,  True,  3, False),
    ]

    ref_auc = None
    for name, ghost, order, k, heal in component_variants:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = build_glassbox(use_ghost=ghost, use_order_decomp=order, n_sub_chunks=k)
        r = run_variant(name, model, X_train, y_train,
                        X_val, y_val, X_test, y_test, cw, heal=heal)
        if ref_auc is None:
            ref_auc = r['test_auc']
        else:
            r['delta'] = round(r['test_auc'] - ref_auc, 4)
        results.append(r)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("   RESULTS SUMMARY")
    print("=" * 60)

    baseline_names = {
        'Linear Baseline', 'MLP Baseline',
        'GlassboxNet (no healing)', 'Full GlassboxNet + Healing',
    }
    component_names = {
        'Full System (reference)', 'w/o Ghost Gates',
        'w/o Order Decomposition', 'w/o MoSE  (K=1)', 'w/o Self-Healing',
    }

    print("\nBASELINE ABLATION")
    print(f"  {'Model':<35}  {'Val AUC':>8}  {'Test AUC':>9}")
    print("  " + "-" * 58)
    for r in results:
        if r['name'] in baseline_names:
            marker = '  <-- our system' if 'Healing' in r['name'] and 'no' not in r['name'] else ''
            print(f"  {r['name']:<35}  {r['val_auc']:>8.4f}  {r['test_auc']:>9.4f}{marker}")

    print("\nCOMPONENT ABLATION")
    print(f"  {'Model':<35}  {'Val AUC':>8}  {'Test AUC':>9}  {'Delta':>7}")
    print("  " + "-" * 68)
    for r in results:
        if r['name'] in component_names:
            delta_str = f"{r.get('delta', 0):>+7.4f}" if 'delta' in r else '    ---'
            print(f"  {r['name']:<35}  {r['val_auc']:>8.4f}  {r['test_auc']:>9.4f}  {delta_str}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ablation_results.json')
    payload = {
        'variants':           results,
        'reference_test_auc': ref_auc,
        'epochs':             EPOCHS,
        'heal_rounds':        HEAL_ROUNDS,
        'seed':               SEED,
        'feature_shape':      list(X_train.shape),
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == '__main__':
    main()
