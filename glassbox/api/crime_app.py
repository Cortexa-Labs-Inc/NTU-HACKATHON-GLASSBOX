"""
CrimeVisionGlassbox FastAPI.

Start:  python3 -m uvicorn api.crime_app:app --reload --port 8001

Requires:
  artefacts/crime_vision.pt              — full model (extractor + Glassbox)
  artefacts/crime_meta.json             — chunk config, class names, metrics
  artefacts/crime_train_features.npz    — cached CNN features for self-heal

Endpoints
---------
GET  /health               — liveness check
GET  /model_info           — chunk config, class names, current metrics
POST /predict              — upload PNG → class + chunk blame + proximity warning
POST /predict_features     — raw feature vector → predict (for batch API use)
GET  /failure_report       — current failure cluster map
GET  /perturbation_recipe  — per-cluster synthetic data generation instructions
POST /self_heal            — trigger self-healing rounds
GET  /heal_history         — round-by-round healing progress
"""

import io
import os
import sys
import json
import numpy as np
import torch
from PIL import Image
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from crime.feature_extractor  import CrimeVisionGlassbox
from crime.image_loader       import get_transforms
from crime.failure_detector   import FailureModeDetector
from crime.perturber          import GaussianPerturber
from crime.self_heal          import SelfHealingLoop
from crime.temporal_smoother  import TemporalSmoother
from crime.temporal_lstm      import TemporalLSTMHead

app = FastAPI(
    title="CrimeVisionGlassbox API",
    description=(
        "Self-healing interpretable anomaly detection from surveillance frames. "
        "TinyCNN multi-scale feature extraction + GlassboxNetV2 interpretability + "
        "Gaussian perturbation self-healing on the CUHK Avenue dataset."
    ),
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

ARTEFACT_DIR = os.path.join(os.path.dirname(__file__), '..', 'artefacts')

# ── Global state ───────────────────────────────────────────────────────────────
_model:     CrimeVisionGlassbox | None = None
_meta:      dict = {}
_transform  = None
_detector:  FailureModeDetector | None = None
_healer:    SelfHealingLoop | None = None
_X_train    = None
_y_train    = None
_X_val      = None
_y_val      = None
_heal_history: list = []
_alert_log:      list  = []
_alert_counter:  int   = 0
_alert_threshold: float = 0.70
_smoother:    TemporalSmoother  = TemporalSmoother(window=8, alpha=0.4)
_lstm_head:   TemporalLSTMHead | None = None
_lstm_meta:   dict = {}


def _load_artefacts():
    global _model, _meta, _transform, _detector
    global _X_train, _y_train, _X_val, _y_val
    global _lstm_head, _lstm_meta

    meta_path  = os.path.join(ARTEFACT_DIR, 'crime_meta.json')
    model_path = os.path.join(ARTEFACT_DIR, 'crime_vision.pt')
    feat_path  = os.path.join(ARTEFACT_DIR, 'crime_train_features.npz')

    if not os.path.exists(meta_path):
        print("[crime_app] No crime_meta.json — run training/crime_train.py first.")
        return

    with open(meta_path) as f:
        _meta = json.load(f)

    cfg = _meta.get('cfg', {})
    _model = CrimeVisionGlassbox(
        n_classes=_meta['n_classes'],
        proj_dim=_meta.get('proj_dim', 32),
        embed_dim=cfg.get('embed_dim', 16),
        backbone=cfg.get('backbone', 'tiny'),
        pretrained=False,                   # weights loaded below
        freeze_backbone=cfg.get('freeze_backbone', False),
        use_ghost=cfg.get('use_ghost', True),
        use_order_decomp=cfg.get('use_order_decomp', True),
        n_sub_chunks=cfg.get('n_sub_chunks', 1),
    )
    if os.path.exists(model_path):
        _model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("[crime_app] Model loaded.")
    else:
        print("[crime_app] No model weights — predictions will be random.")

    _model.eval()
    _transform = get_transforms(image_size=_meta.get('image_size', 64), augment=False)

    # Load LSTM temporal head (optional — graceful if missing)
    lstm_path = os.path.join(ARTEFACT_DIR, 'temporal_lstm.pt')
    lstm_meta_path = os.path.join(ARTEFACT_DIR, 'temporal_lstm_meta.json')
    if os.path.exists(lstm_path) and os.path.exists(lstm_meta_path):
        with open(lstm_meta_path) as f:
            _lstm_meta = json.load(f)
        _lstm_head = TemporalLSTMHead(
            n_features=_lstm_meta['n_features'],
            hidden_dim=_lstm_meta['hidden_dim'],
            n_layers=_lstm_meta['n_layers'],
            n_classes=_lstm_meta['n_classes'],
        )
        _lstm_head.load_state_dict(torch.load(lstm_path, map_location='cpu'))
        _lstm_head.eval()
        _lstm_head.reset_state()
        print(f"[crime_app] LSTM head loaded (val_auc={_lstm_meta.get('val_auc')})")

    # Re-create smoother with correct anomaly class index now that meta is loaded
    global _smoother
    anomaly_idx = next(
        (i for i, n in enumerate(_meta.get('class_names', [])) if 'anomaly' in n.lower()),
        1,
    )
    _smoother = TemporalSmoother(window=8, alpha=0.4, anomaly_class=anomaly_idx)

    if os.path.exists(feat_path):
        d = np.load(feat_path)
        _X_train = d['X_train']
        _y_train = d['y_train']
        _X_val   = d['X_val']
        _y_val   = d['y_val']

    # Load pre-computed heal history from training run
    global _heal_history
    train_hist_path = os.path.join(ARTEFACT_DIR, 'crime_training_history.json')
    if os.path.exists(train_hist_path):
        with open(train_hist_path) as f:
            _heal_history = json.load(f).get('healing', [])
        print(f"[crime_app] Loaded {len(_heal_history)} healing rounds from training history.")

    # Pre-build failure map on val features
    if _model is not None and _X_val is not None:
        _detector = FailureModeDetector(
            _model.glassbox,
            class_names=_meta.get('class_names', []),
            chunk_names=_meta.get('chunk_names', []),
        )
        X_f, y_f, yp_f = _detector.collect_failures(_X_val, _y_val)
        if len(X_f) > 0:
            _detector.fit(X_f, y_f, yp_f,
                          n_clusters=_meta.get('cfg', {}).get('heal_n_clusters', 5))
        print(f"[crime_app] Failure map: {len(X_f)} val failures in "
              f"{len(_detector.cluster_stats)} clusters.")


@app.on_event("startup")
def startup():
    _load_artefacts()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_model():
    if _model is None:
        raise HTTPException(503, "Model not loaded. Run training/crime_train.py first.")


def _class_name(idx: int) -> str:
    names = _meta.get('class_names', [])
    return names[idx] if 0 <= idx < len(names) else f'Class{idx}'


def _image_to_tensor(file_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    return _transform(img).unsqueeze(0)   # (1, 3, H, W)


def _predict_from_image_tensor(x_img: torch.Tensor, true_class: int = None) -> dict:
    _model.eval()
    with torch.no_grad():
        logits, audit = _model(x_img, return_audit=True)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred  = int(np.argmax(probs))

    blame_scores = {}
    if true_class is not None and true_class != pred:
        blame = _model.get_class_pair_contributions(x_img, pred, true_class)
        blame_scores = {k: round(v['blame'], 4) for k, v in blame.items()}

    # Proximity check in CNN feature space
    proximity = {}
    if _detector and _detector.cluster_stats:
        feats = _model.extract(x_img).numpy().flatten()
        idx, dist = _detector.nearest_cluster(feats)
        if idx >= 0:
            stats = _detector.cluster_stats[idx]
            attr  = _detector.attribute_cluster(idx)
            proximity = {
                'nearest_cluster':  idx,
                'distance':         round(dist, 4),
                'warning':          dist < 3.0,
                'cluster_failures': stats['n_samples'],
                'cluster_error':    (f"predicted '{stats['pred_name']}' "
                                     f"→ truth '{stats['true_name']}'"),
                'dominant_chunk':   attr['dominant_chunk'] if attr else 'Unknown',
                'message': (
                    f"Frame near Failure Cluster #{idx} (dist={dist:.2f}). "
                    f"Dominant chunk: {attr['dominant_chunk'] if attr else '?'}. "
                    f"Patch: generate '{stats['true_name']}' frames near centroid."
                ) if dist < 3.0 else None,
            }

    temporal = _smoother.update(probs)
    smoothed_names = {
        _class_name(i): round(float(p), 4)
        for i, p in enumerate(temporal['smoothed_probs'])
    }

    # LSTM temporal prediction (learned sequence model)
    lstm_result = {}
    if _lstm_head is not None:
        feats = _model.extract(x_img).numpy().flatten()
        lstm_pred, lstm_conf = _lstm_head.step(feats)
        lstm_result = {
            'lstm_pred':        _class_name(lstm_pred),
            'lstm_confidence':  round(lstm_conf, 4),
            'val_auc':          _lstm_meta.get('val_auc'),
            'note': 'Learned LSTM over 8-frame sequences (val AUC 0.990 vs frame-level 0.920)',
        }

    # ── Auto-alert ──────────────────────────────────────────────────────────────
    global _alert_counter
    is_anomaly = 'anomaly' in _class_name(pred).lower()
    alert_triggered = is_anomaly and float(probs[pred]) >= _alert_threshold
    if alert_triggered:
        _alert_counter += 1
        contribs = audit.get('chunk_contributions', {})
        dominant_chunk = (
            max(contribs, key=lambda k: abs(contribs[k].get('pred_push',
                                             contribs[k].get('disease_push', 0))))
            if contribs else None
        )
        _alert_log.append({
            'id':               _alert_counter,
            'timestamp':        datetime.now(timezone.utc).isoformat(),
            'predicted_class':  _class_name(pred),
            'confidence':       round(float(probs[pred]), 4),
            'dominant_chunk':   dominant_chunk,
            'proximity_cluster': proximity.get('nearest_cluster') if proximity.get('warning') else None,
        })
        if len(_alert_log) > 100:
            _alert_log.pop(0)

    return {
        'prediction':           pred,
        'predicted_class':      _class_name(pred),
        'confidence':           round(float(probs[pred]), 4),
        'alert_triggered':      alert_triggered,
        'alert_threshold':      _alert_threshold,
        'class_probabilities':  {_class_name(i): round(float(p), 4)
                                  for i, p in enumerate(probs)},
        'temporal': {
            'smoothed_pred':       _class_name(temporal['smoothed_pred']),
            'smoothed_confidence': temporal['smoothed_confidence'],
            'smoothed_probs':      smoothed_names,
            'anomaly_streak':      temporal['anomaly_streak'],
            'window_size':         temporal['window_size'],
            'is_stable':           temporal['is_stable'],
            'note': (
                f"Anomaly confirmed over {temporal['anomaly_streak']} consecutive frames."
                if temporal['anomaly_streak'] >= 3
                else 'Smoothing window building — hold judgment.'
                if not temporal['is_stable']
                else None
            ),
        },
        'chunk_contributions':  audit.get('chunk_contributions', {}),
        'order_decomp':         audit.get('order_decomp', {}),
        'sub_chunk_routing':    audit.get('sub_chunk_routing', {}),
        'lstm_temporal':        lstm_result,
        'blame_scores':         blame_scores,
        'ghost_signals':        {k: round(float(v), 4)
                                  for k, v in audit.get('ghost_signals', {}).items()},
        'proximity_warning':    proximity,
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        'status':       'ok',
        'model_loaded': _model is not None,
        'failure_map':  bool(_detector and _detector.cluster_stats),
        'n_clusters':   len(_detector.cluster_stats) if _detector else 0,
    }


@app.get("/model_info")
def model_info():
    if not _meta:
        raise HTTPException(503, "No metadata — run training/crime_train.py first.")
    val_auc  = _meta.get('best_val_auc')
    test_auc = _meta.get('test_auc')
    gap_note = None
    if val_auc and test_auc:
        gap = round(val_auc - test_auc, 4)
        gap_note = (
            f"Val AUC {val_auc:.3f} vs Test AUC {test_auc:.3f} (gap={gap:+.3f}). "
            "Self-healing optimises on the val set — test AUC is the honest held-out number."
        )

    return {
        'chunk_names':   _meta.get('chunk_names', []),
        'chunk_sizes':   _meta.get('chunk_sizes', []),
        'n_chunks':      len(_meta.get('chunk_names', [])),
        'n_classes':     _meta.get('n_classes', 0),
        'class_names':   _meta.get('class_names', []),
        'n_features':    _meta.get('n_features', 0),
        'proj_dim':      _meta.get('proj_dim', 32),
        'image_size':    _meta.get('image_size', 64),
        'test_auc':      test_auc,
        'test_acc':      _meta.get('test_acc'),
        'best_val_auc':  val_auc,
        'n_train':       _meta.get('n_train'),
        'architecture':  'TinyCNN(4-stage) → N-chunk GlassboxNetV2 (Ghost gates + order decomp)',
        'training_domain': {
            'dataset':   'CUHK Avenue',
            'scene':     'Fixed university walkway camera',
            'note':      (
                'Model was trained on a single fixed-camera scene. '
                'Accuracy will degrade on cameras with different angles, '
                'lighting, or scene types — retrain on domain-specific footage for production use.'
            ),
        },
        'chunk_design_note': (
            f"Chunks ({len(_meta.get('chunk_names', []))}) are aligned to CNN stages "
            "(Texture/Structure/Context/Semantic). Count is a design choice for interpretability — "
            "n_chunks is configurable at training time. More chunks increase ghost gate pairs "
            "quadratically (C(n,2)) and make attribution less readable."
        ),
        'auc_gap_note':  gap_note,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    true_class: Optional[int] = None,
):
    """
    Upload a PNG surveillance frame → crime prediction + full interpretability.

    Returns:
      predicted_class       : crime category name
      chunk_contributions   : which CNN scale (Texture/Structure/Context/Semantic) drove prediction
      order_decomp          : was each chunk's decision linear or nonlinear?
      ghost_signals         : cross-scale interaction gate activity (C0→C1, ...)
      blame_scores          : (if true_class provided) per-chunk blame for misclassification
      proximity_warning     : is this frame near a known failure cluster?
    """
    _require_model()
    raw = await file.read()
    try:
        x_img = _image_to_tensor(raw)
    except Exception as e:
        raise HTTPException(400, f"Cannot read image: {e}")
    return _predict_from_image_tensor(x_img, true_class=true_class)


class FeaturePredictRequest(BaseModel):
    features:   list[float]
    true_class: Optional[int] = None


@app.post("/predict_features")
def predict_features(req: FeaturePredictRequest):
    """Predict directly from pre-extracted CNN feature vector (4 × proj_dim)."""
    _require_model()
    n_feat = _meta.get('n_features', 128)
    if len(req.features) != n_feat:
        raise HTTPException(400, f"Expected {n_feat} features, got {len(req.features)}.")

    x_feat = torch.FloatTensor(req.features).unsqueeze(0)
    _model.eval()
    with torch.no_grad():
        logits, audit = _model.glassbox(x_feat, return_audit=True)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred  = int(np.argmax(probs))

    blame_scores = {}
    if req.true_class is not None and req.true_class != pred:
        blame = _model.glassbox.get_class_pair_contributions(x_feat, pred, req.true_class)
        blame_scores = {k: round(v['blame'], 4) for k, v in blame.items()}

    return {
        'prediction':          pred,
        'predicted_class':     _class_name(pred),
        'confidence':          round(float(probs[pred]), 4),
        'class_probabilities': {_class_name(i): round(float(p), 4)
                                 for i, p in enumerate(probs)},
        'chunk_contributions': audit.get('chunk_contributions', {}),
        'order_decomp':        audit.get('order_decomp', {}),
        'blame_scores':        blame_scores,
        'ghost_signals':       {k: round(float(v), 4)
                                 for k, v in audit.get('ghost_signals', {}).items()},
    }


@app.get("/failure_report")
def failure_report():
    _require_model()
    if _detector is None or not _detector.cluster_stats:
        return {'status': 'no_failures', 'clusters': []}
    return _detector.get_failure_report()


@app.get("/perturbation_recipe")
def perturbation_recipe():
    _require_model()
    if _detector is None or not _detector.cluster_stats:
        return {'status': 'no_failures', 'recipes': []}
    sigma     = _meta.get('cfg', {}).get('heal_sigma', 0.3)
    perturber = GaussianPerturber(sigma_scale=sigma)
    recipes   = perturber.get_perturbation_recipe(
        _detector.cluster_stats, class_names=_meta.get('class_names', [])
    )
    return {
        'status':     'ok',
        'n_clusters': len(recipes),
        'note': (
            'Perturbation is in CNN feature space (4 × proj_dim). '
            'Synthetic feature vectors teach the Glassbox to correctly '
            'classify near these failure centroids without image synthesis.'
        ),
        'recipes': recipes,
    }


class SelfHealRequest(BaseModel):
    max_rounds:  Optional[int]   = 3
    n_clusters:  Optional[int]   = 5
    n_synthetic: Optional[int]   = 50
    sigma_scale: Optional[float] = 0.3


@app.post("/self_heal")
def self_heal(req: SelfHealRequest):
    global _healer, _heal_history, _detector
    _require_model()
    if _X_train is None:
        raise HTTPException(
            503, "Cached features missing. Ensure crime_train_features.npz exists."
        )

    _healer = SelfHealingLoop(
        model=_model.glassbox,
        X_train=_X_train,
        y_train=_y_train,
        X_val=_X_val,
        y_val=_y_val,
        class_names=_meta.get('class_names', []),
        chunk_names=_meta.get('chunk_names', []),
        n_clusters=req.n_clusters,
        n_synthetic=req.n_synthetic,
        sigma_scale=req.sigma_scale,
        epochs_per_round=20,
        max_rounds=req.max_rounds,
        patience=2,
    )
    history = _healer.run()
    _heal_history = history

    _detector = FailureModeDetector(
        _model.glassbox,
        class_names=_meta.get('class_names', []),
        chunk_names=_meta.get('chunk_names', []),
    )
    X_f, y_f, yp_f = _detector.collect_failures(_X_val, _y_val)
    if len(X_f) > 0:
        _detector.fit(X_f, y_f, yp_f, n_clusters=req.n_clusters)

    return _healer.get_summary()


@app.get("/heal_history")
def heal_history():
    if not _heal_history:
        return {'status': 'not_run', 'history': []}
    return {'status': 'ok', 'history': _heal_history}


@app.get("/per_video_validation")
def per_video_validation():
    """
    Per-video AUC breakdown across all 21 CUHK Avenue test videos.
    Each video = different temporal scene context (equivalent to different camera angles
    within the same installation). Mean AUC 0.965 ± 0.039 across 20 mixed-class videos.
    """
    val_path = os.path.join(ARTEFACT_DIR, 'per_video_validation.json')
    if not os.path.exists(val_path):
        return {'status': 'not_generated'}
    with open(val_path) as f:
        return json.load(f)


@app.get("/sub_chunk_profiles")
def sub_chunk_profiles():
    """
    Return auto-generated profiles of what each sub-expert within each named chunk
    has learned to detect, derived from routing weight analysis over training data.
    """
    profile_path = os.path.join(ARTEFACT_DIR, 'subchunk_profiles.json')
    if not os.path.exists(profile_path):
        return {'status': 'not_generated',
                'note': 'Run crime/subchunk_profiler.py to generate profiles.'}
    with open(profile_path) as f:
        profiles = json.load(f)
    return {
        'status':   'ok',
        'n_chunks': len(profiles),
        'profiles': profiles,
        'note': (
            'Sub-expert labels are auto-generated from routing weight analysis over '
            'training data. anomaly_lift = P(Anomaly | sub-expert top-20 activated) '
            '− base_rate. Positive lift → sub-expert responds to anomalous patterns.'
        ),
    }


@app.post("/reset_temporal")
def reset_temporal():
    """Reset EMA smoother + LSTM state (call when switching camera streams)."""
    _smoother.reset()
    if _lstm_head is not None:
        _lstm_head.reset_state()
    return {'status': 'ok', 'message': 'Temporal buffer and LSTM state cleared.'}


# ── Alert system ───────────────────────────────────────────────────────────────

@app.get("/alerts")
def get_alerts(limit: int = 20):
    """
    Return the in-memory alert log (most recent first).
    Alerts are auto-generated when an Anomaly prediction exceeds the confidence threshold.
    """
    return {
        'threshold':  _alert_threshold,
        'total':      len(_alert_log),
        'alerts':     list(reversed(_alert_log[-limit:])),
    }


class AlertThresholdRequest(BaseModel):
    threshold: float


@app.post("/alerts/threshold")
def set_alert_threshold(req: AlertThresholdRequest):
    """Update the confidence threshold that triggers an alert (0.0–1.0)."""
    global _alert_threshold
    if not 0.0 <= req.threshold <= 1.0:
        raise HTTPException(400, "Threshold must be between 0.0 and 1.0.")
    _alert_threshold = req.threshold
    return {'status': 'ok', 'threshold': _alert_threshold}


@app.post("/alerts/clear")
def clear_alerts():
    """Clear the in-memory alert log."""
    global _alert_log, _alert_counter
    _alert_log.clear()
    _alert_counter = 0
    return {'status': 'ok', 'message': 'Alert log cleared.'}
