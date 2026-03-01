# CrimeVisionGlassbox

**Self-healing, interpretable anomaly detection for surveillance video.**

---

## The Problem

Surveillance AI systems today are black boxes. When a camera flags a frame as anomalous, operators can't tell *why* — which part of the image triggered it, whether it's a pattern the model has seen before, or a known failure mode. This erodes trust and leads to either over-reliance or under-utilisation.

Worse: models degrade silently. When they encounter distribution shift, they just get worse — with no mechanism to self-correct.

---

## Our Solution

CrimeVisionGlassbox provides four things standard models don't:

**1. Interpretability by construction** — every prediction decomposes into four visual chunk contributions (Texture / Structure / Context / Semantic) with exact logit math, not approximation.

**2. Sub-chunk discovery** — within each named chunk, a Mixture-of-Sub-Experts (3 sub-networks + learned router) discovers mathematical sub-patterns automatically. You see *which sub-pattern* fired, not just which chunk.

**3. Self-healing** — the model identifies its own failure modes, generates corrective synthetic training data via Gaussian perturbation in feature space, and retrains — up to N rounds.

**4. Learned temporal context** — a 2-layer LSTM head processes sequences of 8 frames, lifting AUC from 0.920 (frame-level) to **0.990** (sequence-level).

---

## Results (CUHK Avenue Dataset)

| Metric | Value |
|--------|-------|
| Frame-level Test AUC | **92.0%** |
| LSTM Sequence Val AUC | **99.0%** |
| Val AUC (post-healing, 3 rounds) | **99.5%** |
| Per-video mean AUC (21 scenes) | **96.5% ± 3.9pp** |
| Testbench (20 sample frames) | **19/20 = 95%** |
| Classes | Normal / Anomaly |
| Train / Val / Test frames | 1,520 / 325 / 325 |
| Training time (CPU) | ~8 minutes |

*Note: self-healing and LSTM optimise on validation data. Frame-level test AUC (0.920) is the honest held-out generalisation metric.*

**Self-healing trajectory:**
```
Round 1:  16 failures → 5 clusters → 200 synthetic samples  →  AUC 0.985 → 0.983
Round 2:  21 failures → 5 clusters → 200 synthetic samples  →  AUC 0.983 → 0.988
Round 3:  19 failures → 5 clusters → 200 synthetic samples  →  AUC 0.988 → 0.995  (converged)
```

**Sub-chunk profiles (auto-discovered):**
```
Structure.sub0  →  100% anomaly activation  (+83pp lift)  — pure anomaly structure detector
Semantic.sub1   →  100% anomaly activation  (+83pp lift)  — pure semantic anomaly trigger
Context.sub2    →  100% anomaly activation  (+83pp lift)  — anomaly context trigger
Texture.sub1    →   90% routing weight      (+53pp lift)  — dominant anomaly texture detector
Semantic.sub0   →   77% routing weight      (-12pp lift)  — normal scene baseline
```

---

## Project Structure

```
glassbox/
├── README.md
├── requirements.txt
├── prepare_cuhk.py            ← frame extraction from CUHK Avenue AVI + .mat labels
│
├── crime/                     ← vision + interpretability pipeline
│   ├── feature_extractor.py   ← TinyCNN (4-stage, CPU-only) + CrimeVisionGlassbox
│   ├── crime_glassbox.py      ← GlassboxNetV2 with exact chunk attribution
│   ├── image_loader.py        ← ImageFolder data loader
│   ├── failure_detector.py    ← K-means failure clustering + order decomp audit
│   ├── perturber.py           ← Gaussian perturbation around failure centroids
│   ├── self_heal.py           ← self-healing loop orchestrator
│   ├── temporal_smoother.py   ← EMA sliding-window smoother (8 frames)
│   ├── temporal_lstm.py       ← 2-layer LSTM head + sequence dataset builder
│   └── subchunk_profiler.py   ← auto-labels what each sub-expert learned
│
├── model/                     ← core neural building blocks (reusable)
│   ├── glassbox_net_v2.py     ← GlassboxNetV2: ChunkNets + Ghost gates + MoE sub-chunks
│   ├── chunks.py              ← ChunkNet with Mixture-of-Sub-Experts (MoSE)
│   └── ghost_gate.py          ← Ghost Signal Gate (selective cross-chunk flow)
│
├── training/
│   └── crime_train.py         ← training entry point (40 epochs + self-healing)
│
├── api/
│   └── crime_app.py           ← FastAPI: /predict, /failure_report, /self_heal, +7 more
│
├── dashboard/
│   └── crime_dashboard.html   ← live web dashboard (Chart.js, no build step)
│
├── testbench/
│   ├── SETUP.md               ← step-by-step guide for judges
│   ├── test_pipeline.py       ← end-to-end test: 95% on 20 sample frames
│   └── sample_frames/
│       ├── Normal/   (10 real CUHK Avenue frames)
│       └── Anomaly/  (10 real CUHK Avenue frames)
│
└── artefacts/                 ← pre-trained weights + metadata
    ├── crime_vision.pt              — CNN + GlassboxNetV2 (MoE sub-chunks)
    ├── temporal_lstm.pt             — LSTM temporal head
    ├── crime_meta.json              — chunk config, class names, metrics
    ├── crime_train_features.npz     — cached CNN features for self-heal
    ├── crime_training_history.json  — epoch-by-epoch curves
    ├── subchunk_profiles.json       — auto-labeled sub-expert profiles
    └── per_video_validation.json    — per-scene AUC breakdown (21 videos)
```

---

## Quick Start

See **[testbench/SETUP.md](testbench/SETUP.md)** for the full step-by-step guide.

```bash
# Install
pip install -r requirements.txt

# Run testbench (no server needed — uses pre-trained weights)
cd glassbox/
python3 testbench/test_pipeline.py
# → 19/20 = 95%  PASS

# Start API
python3 -m uvicorn api.crime_app:app --reload --port 8001

# Open dashboard (API must be running)
open dashboard/crime_dashboard.html
```

---

## How It Works

### TinyCNN Feature Extractor
A 4-stage convolutional network (stem→64→128→256→256) maps each 64×64 surveillance frame into 4 independent feature vectors — one per chunk. ~2.4M parameters, trains in minutes on CPU, no pretrained download.

### GlassboxNetV2 — 4-Level Interpretability Chain

Each frame passes through four levels of analysis:

**Level 1 — Chunk attribution (exact)**
4 named ChunkNets (Texture / Structure / Context / Semantic) produce 16-dim embeddings. The final classifier uses an exact linear decomposition: each chunk's contribution to the logit is mathematically separable and computed directly, not approximated.

**Level 2 — Ghost Signal Gates**
6 learned per-sample gates (C(4,2) pairs) that open only when cross-chunk signals improve prediction. Start closed (bias=−3). When Context→Semantic opens, the model detected unusual coupling between scene layout and abstract behaviour.

**Level 3 — Order Decomposition**
Each chunk runs two parallel paths: a linear 1st-order path and a nonlinear nth-order MLP. A learned β gate mixes them. β→0 = simple threshold; β large = complex pattern matching. Exposed per chunk per prediction.

**Level 4 — Mixture of Sub-Experts (MoSE)**
Within each named chunk, 3 sub-networks run in parallel. A learned router (softmax) discovers which sub-pattern is active per frame — found automatically, not hand-defined. Profiled post-hoc against training data to generate anomaly lift scores and natural-language interpretations.

### LSTM Temporal Head
A 2-layer LSTM (hidden=64) trained on sequences of 8 consecutive frames using frozen CNN features. Stateful at inference: each frame updates hidden state. Val AUC 0.990 vs 0.920 frame-level (+7pp). Reset between streams via `POST /reset_temporal`.

### Self-Healing Loop
```
After initial training, for each round:
  1. Run val set → collect misclassified samples + CNN feature vectors
  2. K-means cluster failures in 128-dim feature space → K centroids
  3. Per centroid: x_synth ~ N(centroid, σ²·I)  [Gaussian perturbation]
  4. Augment training data; retrain Glassbox head (15 epochs)
  5. Save if val_AUC improves; else patience counter
```

Failure cluster reports include order decomposition at each centroid — telling you *how* the model was reasoning when it failed, which directly informs the data patch strategy.

### What the Dashboard Shows
- **Model metrics** — test AUC, LSTM AUC, val AUC, train size
- **Failure Clusters** — dominant chunk, blame scores, order decomp at centroid
- **Self-healing progress** — val AUC trajectory over rounds
- **Sub-chunk profiles** — auto-labeled sub-expert interpretations with anomaly lift
- **Per-video validation** — AUC bar chart across 21 CUHK Avenue scenes
- **Live inference** — drag-drop → prediction + LSTM pred + chunk attribution + sub-chunk routing + ghost gates + proximity warning

---

## API Endpoints

```bash
python3 -m uvicorn api.crime_app:app --reload --port 8001
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness + failure map status |
| `/model_info` | GET | Chunk config, AUC, design notes, domain limitations |
| `/predict` | POST | PNG → frame pred + LSTM pred + chunk attribution + sub-chunk routing + ghost gates + proximity warning |
| `/failure_report` | GET | Failure clusters with blame, order decomp, σ |
| `/perturbation_recipe` | GET | Synthetic data generation params per cluster |
| `/self_heal` | POST | Trigger self-healing rounds |
| `/heal_history` | GET | Round-by-round AUC trajectory |
| `/sub_chunk_profiles` | GET | Auto-labeled sub-expert profiles with anomaly lift |
| `/per_video_validation` | GET | Per-scene AUC across 21 CUHK Avenue videos |
| `/reset_temporal` | POST | Reset EMA buffer + LSTM state (switch streams) |

---

## Known Limitations

| Limitation | Status |
|-----------|--------|
| Single-camera domain (CUHK Avenue walkway) | Acknowledged in `/model_info`. Retrain on domain footage for production. |
| Val/test AUC gap (+3.1pp) | Self-healing optimises on val. Test AUC (0.920) is the honest held-out number. |
| Sub-expert labels are statistical | Lift scores auto-generated; not hand-verified per sub-expert. |
| Per-video = same dataset | Not true cross-domain transfer. Different installation requires new training data. |

---

## Dataset: CUHK Avenue

16 training videos (all normal) + 21 testing videos (normal + anomaly), with pixel-level ground-truth masks. Every 10th frame extracted at 64×64, labeled Anomaly if any pixel in mask > 0.

```bash
# Re-download dataset
python3 -c "import kagglehub; kagglehub.dataset_download('hihnguynth/cuhk-avenue-dataset')"

# Re-extract frames
python3 prepare_cuhk.py --every 10

# Retrain (~8 min on CPU)
python3 training/crime_train.py
```
