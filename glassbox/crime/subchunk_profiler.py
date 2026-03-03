"""
SubChunkProfiler — automatically label what each sub-expert captures.

For each named chunk C and each sub-expert k:
  1. Run all training frames through the model
  2. Collect routing weight w_{c,k} per frame
  3. Top-20 most activated frames → their true class, mean feature values
  4. Compute:
       - dominant_class   : which class activates this sub-expert most
       - class_bias       : P(Anomaly | sub-expert active) vs base rate
       - top_feature_dims : which input dims are highest in top-activated frames
       - label            : auto-generated description string

The profiles are saved to artefacts/subchunk_profiles.json and served via
GET /sub_chunk_profiles.
"""

import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image


class SubChunkProfiler:
    """
    Profiles sub-chunk routing patterns against training data.

    Parameters
    ----------
    model      : CrimeVisionGlassbox  (full model with CNN + Glassbox)
    chunk_names: list of human-readable chunk names
    n_sub_chunks: number of sub-experts per chunk
    transform  : torchvision transform for loading images
    top_k      : number of top-activated frames per sub-expert (default 20)
    """

    def __init__(self, model, chunk_names: list, n_sub_chunks: int,
                 transform, top_k: int = 20):
        self.model        = model
        self.chunk_names  = chunk_names
        self.n_sub_chunks = n_sub_chunks
        self.transform    = transform
        self.top_k        = top_k
        self.profiles     = {}   # populated by run()

    def run(self, data_dir: str, class_names: list) -> dict:
        """
        Scan all images in data_dir/{class_name}/ subdirs, collect routing weights,
        and build per-sub-expert profiles.

        Returns profiles dict.
        """
        data_path = Path(data_dir)
        all_frames = []   # list of (img_path, label_idx, label_name)

        for label_idx, label_name in enumerate(class_names):
            class_dir = data_path / label_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob('*.png')):
                all_frames.append((img_path, label_idx, label_name))

        if not all_frames:
            return {}

        print(f"[SubChunkProfiler] Profiling {len(all_frames)} frames "
              f"across {self.n_sub_chunks} sub-experts × {len(self.chunk_names)} chunks ...")

        # ── Collect routing weights per frame ────────────────────────────────
        # routing_weights[chunk_idx][sub_idx] = list of (weight, label, path)
        routing_records = {
            c: {k: [] for k in range(self.n_sub_chunks)}
            for c in range(len(self.chunk_names))
        }

        self.model.eval()
        with torch.no_grad():
            for img_path, label_idx, label_name in all_frames:
                try:
                    img    = Image.open(img_path).convert('RGB')
                    tensor = self.transform(img).unsqueeze(0)
                    _, audit = self.model(tensor, return_audit=True)
                    routing = audit.get('sub_chunk_routing', {})

                    for c_idx, cname in enumerate(self.chunk_names):
                        info = routing.get(cname, {})
                        weights = info.get('routing_weights', {})
                        for k in range(self.n_sub_chunks):
                            w = weights.get(f'sub{k}', 0.0)
                            routing_records[c_idx][k].append(
                                (w, label_idx, label_name, str(img_path.name))
                            )
                except Exception:
                    continue

        # ── Build profiles ───────────────────────────────────────────────────
        anomaly_idx = class_names.index('Anomaly') if 'Anomaly' in class_names else -1
        n_anomaly = sum(1 for _, li, _ in all_frames if li == anomaly_idx)
        base_anomaly_rate = n_anomaly / max(len(all_frames), 1)

        profiles = {}
        for c_idx, cname in enumerate(self.chunk_names):
            profiles[cname] = {}
            for k in range(self.n_sub_chunks):
                records = routing_records[c_idx][k]
                records_sorted = sorted(records, key=lambda r: r[0], reverse=True)
                top = records_sorted[:self.top_k]

                if not top:
                    continue

                top_weights    = [r[0] for r in top]
                top_labels     = [r[2] for r in top]
                top_fnames     = [r[3] for r in top]
                mean_weight    = float(np.mean([r[0] for r in records]))

                # Class bias
                anomaly_class  = 'Anomaly' if 'Anomaly' in class_names else class_names[1]
                n_anom_top     = sum(1 for l in top_labels if l == anomaly_class)
                anomaly_frac   = n_anom_top / max(len(top), 1)
                dominant_class = anomaly_class if anomaly_frac > 0.5 else \
                                 (class_names[0] if class_names else 'Normal')

                # Lift vs base rate
                lift = anomaly_frac - base_anomaly_rate

                # Auto-generate label
                label = _auto_label(cname, k, dominant_class, anomaly_frac,
                                    base_anomaly_rate, mean_weight)

                profiles[cname][f'sub{k}'] = {
                    'label':          label,
                    'dominant_class': dominant_class,
                    'anomaly_frac':   round(anomaly_frac, 3),
                    'anomaly_lift':   round(lift, 3),
                    'mean_routing':   round(mean_weight, 4),
                    'top_k_samples':  top_fnames[:5],   # top-5 filenames
                    'top_k_weights':  [round(w, 4) for w in top_weights[:5]],
                    'interpretation': _interpret(anomaly_frac, base_anomaly_rate,
                                                 mean_weight, dominant_class, cname),
                }

        self.profiles = profiles
        return profiles


def _auto_label(chunk_name: str, sub_idx: int, dominant_class: str,
                anomaly_frac: float, base_rate: float, mean_weight: float) -> str:
    """Generate a short human-readable label for a sub-expert."""
    lift = anomaly_frac - base_rate
    specificity = 'highly specific' if mean_weight > 0.7 else \
                  'moderately active' if mean_weight > 0.4 else 'broadly active'

    if abs(lift) < 0.05:
        bias = 'class-neutral'
    elif lift > 0.3:
        bias = 'anomaly-biased'
    elif lift > 0.1:
        bias = 'anomaly-leaning'
    elif lift < -0.3:
        bias = 'normal-biased'
    else:
        bias = 'normal-leaning'

    return f"{chunk_name}.sub{sub_idx} — {specificity}, {bias}"


def _interpret(anomaly_frac: float, base_rate: float, mean_weight: float,
               dominant_class: str, chunk_name: str) -> str:
    """One-sentence human interpretation of what this sub-expert does."""
    lift = anomaly_frac - base_rate
    if mean_weight > 0.65:
        activity = f"dominant sub-expert within {chunk_name}"
    elif mean_weight > 0.35:
        activity = f"co-active sub-expert within {chunk_name}"
    else:
        activity = f"rarely selected sub-expert within {chunk_name}"

    if abs(lift) < 0.05:
        sensitivity = "activates equally for both classes — captures class-invariant patterns"
    elif lift > 0.2:
        sensitivity = (f"activates {lift*100:.0f}pp more on Anomaly frames — "
                       f"likely detects unusual {chunk_name.lower()} patterns")
    elif lift > 0.05:
        sensitivity = (f"slightly anomaly-biased (+{lift*100:.0f}pp) — "
                       f"may respond to borderline {chunk_name.lower()} irregularities")
    else:
        sensitivity = (f"primarily activates on Normal frames — "
                       f"likely captures regular {chunk_name.lower()} patterns")

    return f"A {activity} that {sensitivity}."


def run_and_save(model, chunk_names: list, n_sub_chunks: int,
                 transform, data_dir: str, class_names: list,
                 out_path: str, top_k: int = 20) -> dict:
    """Convenience: profile and save to JSON."""
    profiler = SubChunkProfiler(
        model=model,
        chunk_names=chunk_names,
        n_sub_chunks=n_sub_chunks,
        transform=transform,
        top_k=top_k,
    )
    profiles = profiler.run(data_dir, class_names)
    with open(out_path, 'w') as f:
        json.dump(profiles, f, indent=2)
    print(f"[SubChunkProfiler] Saved profiles → {out_path}")
    return profiles
