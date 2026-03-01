# CrimeVisionGlassbox — Setup & Run Guide

Step-by-step instructions for judges to set up, run, and verify the project.

---

## Prerequisites

- Python 3.10 or 3.11 (3.12 also works)
- macOS, Linux, or Windows (WSL recommended on Windows)
- ~2 GB free disk space
- No GPU required — runs fully on CPU

---

## Step 1 — Clone / Open the Repository

```bash
cd glassbox/   # the project root (where this SETUP.md lives one level up)
```

All commands below are run from the **`glassbox/`** directory.

---

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages installed:
- `torch`, `torchvision` — neural network
- `fastapi`, `uvicorn` — REST API
- `pillow`, `opencv-python` — image processing
- `scikit-learn`, `scipy` — clustering and evaluation
- `kagglehub` — dataset download (only needed if retraining)

---

## Step 3 — Verify Trained Model Exists

The repository includes pre-trained artefacts in `artefacts/`. Verify:

```bash
ls artefacts/
# Expected output includes:
#   crime_vision.pt
#   crime_meta.json
#   crime_train_features.npz
#   crime_training_history.json
```

If these files are missing, retrain (see Step 6).

---

## Step 4 — Run the Testbench (No Server Needed)

This is the quickest way to verify everything works:

```bash
python3 testbench/test_pipeline.py
```

**Expected output:**
```
============================================================
  CrimeVisionGlassbox — Testbench
============================================================

Model info:
  Classes:     ['Anomaly', 'Normal']
  Chunks:      ['Texture', 'Structure', 'context', 'Semantic']
  Test AUC:    0.9231
  Test Acc:    0.9231
  Val AUC:     0.9538

Model loaded from crime_vision.pt

Frame                                   True     Pred   Conf  Chunk blame
--------------------------------------------------------------------------------
✓ 01_f000510.png                      Normal   Normal 100.0%  ...
✓ 01_f000640.png                     Anomaly  Anomaly 100.0%  ...
...

============================================================
  Accuracy on 20 sample frames: 19/20 = 95.0%
  PASS (>= 80% threshold)
============================================================
```

The testbench:
- Loads the trained model from `artefacts/crime_vision.pt`
- Runs inference on 20 sample frames (10 Normal + 10 Anomaly) from `testbench/sample_frames/`
- Prints class prediction, confidence, and per-chunk attribution scores
- Checks accuracy >= 80% and prints PASS/WARN

---

## Step 5 — Start the API and Dashboard

### 5a. Start the API server

```bash
python3 -m uvicorn api.crime_app:app --reload --port 8001
```

Wait for: `INFO:     Application startup complete.`

### 5b. Check the API is live

```bash
curl http://localhost:8001/health
# {"status": "ok", "model_loaded": true}

curl http://localhost:8001/model_info
# {"n_classes": 2, "class_names": ["Anomaly", "Normal"], "test_auc": 0.9231, ...}
```

### 5c. Run a prediction via API

```bash
curl -X POST http://localhost:8001/predict \
  -F "file=@testbench/sample_frames/Anomaly/01_f000640.png"
```

Example response:
```json
{
  "predicted_class": "Anomaly",
  "confidence": 1.0,
  "class_probabilities": {"Anomaly": 1.0, "Normal": 0.0},
  "chunk_attributions": {
    "Texture": {"pred_push": -0.86, "true_push": -0.86, "blame": 0.0},
    "Structure": {"pred_push": 0.63, ...},
    "Context": {"pred_push": 4.24, ...},
    "Semantic": {"pred_push": 0.61, ...}
  }
}
```

### 5d. View failure cluster report

```bash
curl http://localhost:8001/failure_report
```

### 5e. Trigger a self-healing round

```bash
curl -X POST http://localhost:8001/self_heal
```

### 5f. Open the live dashboard

Open `dashboard/crime_dashboard.html` in your browser.

Make sure the API is running on port 8001 (the dashboard connects to `http://localhost:8001`).

---

## Step 6 — (Optional) Retrain From Scratch

This takes ~8-10 minutes on CPU. Requires a Kaggle account for dataset download.

### 6a. Download the CUHK Avenue dataset

```bash
python3 -c "
import kagglehub
path = kagglehub.dataset_download('hihnguynth/cuhk-avenue-dataset')
print('Downloaded to:', path)
"
```

> **Note:** `kagglehub` will prompt for Kaggle credentials on first use.
> Accept the dataset license at kaggle.com if prompted.

### 6b. Extract frames

```bash
python3 prepare_cuhk.py --every 10
```

This extracts every 10th frame from 37 videos into `CUHK_Avenue/Normal/` and `CUHK_Avenue/Anomaly/`.
Expected output: ~2718 Normal + ~367 Anomaly frames.

### 6c. Train

```bash
python3 training/crime_train.py
```

Training log example:
```
Classes (2): ['Anomaly', 'Normal']
Split: train=1520, val=325, test=325
Epoch   1 | val_auc=0.8615 | val_acc=0.8615
...
Epoch  40 | val_auc=0.9446 | val_acc=0.9446
Best val AUC: 0.9538
Test AUC: 0.9231  |  Test Acc: 0.9231

Starting self-healing rounds ...
  Round 1 | failures=15 | AUC 0.9873→0.9930
  Round 2 | failures=12 | AUC 0.9930→0.9976
  Round 3 | failures=8  | AUC 0.9976→0.9974

Artefacts saved to artefacts/
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: torch` | Run `pip install -r requirements.txt` |
| `Missing: artefacts/crime_vision.pt` | Run `python3 training/crime_train.py` |
| API returns 500 on `/predict` | Ensure `crime_vision.pt` matches current model architecture |
| Dashboard shows "Cannot connect" | Verify API is running on port 8001 |
| Kaggle download fails | Accept dataset license at kaggle.com/datasets/hihnguynth/cuhk-avenue-dataset |

---

## File Reference

| File | Purpose |
|------|---------|
| `testbench/test_pipeline.py` | **Main test** — run this to verify the system |
| `testbench/sample_frames/` | 20 labeled PNG frames for testing |
| `artefacts/crime_vision.pt` | Trained model weights |
| `artefacts/crime_meta.json` | Model config + performance metrics |
| `training/crime_train.py` | Full training script |
| `prepare_cuhk.py` | Dataset preparation from CUHK Avenue |
| `api/crime_app.py` | FastAPI server |
| `dashboard/crime_dashboard.html` | Live web dashboard |
