"""
Extract frames from CUHK Avenue Dataset and build a binary ImageFolder
structure for crime_train.py.

Output:
  CUHK_Avenue/train/Normal/     ← frames from training_videos (all normal)
  CUHK_Avenue/train/Anomaly/    ← anomaly frames from test_videos 01-16
  CUHK_Avenue/test/Normal/      ← normal frames from test_videos 17-21
  CUHK_Avenue/test/Anomaly/     ← anomaly frames from test_videos 17-21

Labels are derived from the pixel-level ground-truth masks:
  frame is Anomaly if any pixel in its volLabel mask is > 0.

Usage:
  python3 prepare_cuhk.py [--train_every N] [--test_every M]
"""

import argparse
import cv2
import numpy as np
import scipy.io
from pathlib import Path

# ── Dataset paths ──────────────────────────────────────────────────────────────
BASE = Path.home() / '.cache/kagglehub/datasets/hihnguynth/cuhk-avenue-dataset/versions/1'
AVENUE = BASE / 'Avenue_Dataset' / 'Avenue Dataset'
TRAIN_VIDS = AVENUE / 'training_videos'
TEST_VIDS  = AVENUE / 'testing_videos'
LABEL_DIR  = BASE / 'ground_truth_demo' / 'ground_truth_demo' / 'testing_label_mask'

OUT = Path(__file__).parent / 'CUHK_Avenue'

# If flat=True: all frames → CUHK_Avenue/Normal/ and CUHK_Avenue/Anomaly/
# The image_loader then does a random 70/15/15 split automatically.
# This avoids domain shift from grouping videos by split.
FLAT = True


def load_frame_labels(vid_id: int) -> np.ndarray:
    """Load binary frame labels (0=Normal, 1=Anomaly) from ground-truth mask."""
    mat = scipy.io.loadmat(str(LABEL_DIR / f'{vid_id}_label.mat'))
    vol = mat['volLabel']          # shape (1, N_frames), each cell is (H, W) uint8
    n = vol.shape[1]
    labels = np.array([vol[0, i].max() > 0 for i in range(n)], dtype=np.uint8)
    return labels


def extract_video(avi_path: Path, out_normal: Path, out_anomaly: Path | None,
                  frame_labels: np.ndarray | None, every: int,
                  tag: str, frame_offset: int = 0) -> tuple[int, int]:
    """
    Extract every `every`-th frame from a video.
    If frame_labels is None, all frames go to out_normal.
    Returns (n_normal, n_anomaly) written.
    """
    cap = cv2.VideoCapture(str(avi_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_labels is not None and len(frame_labels) != n_frames:
        # Truncate or pad (should not happen with correct data)
        if len(frame_labels) < n_frames:
            frame_labels = np.concatenate([frame_labels,
                np.zeros(n_frames - len(frame_labels), dtype=np.uint8)])
        else:
            frame_labels = frame_labels[:n_frames]

    out_normal.mkdir(parents=True, exist_ok=True)
    if out_anomaly is not None:
        out_anomaly.mkdir(parents=True, exist_ok=True)

    n_normal = n_anomaly = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every == 0:
            # Resize to 64×64
            small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
            fname = f'{tag}_f{frame_offset + idx:06d}.png'
            is_anomaly = (frame_labels is not None) and (frame_labels[idx] > 0)
            if is_anomaly:
                cv2.imwrite(str(out_anomaly / fname), small)
                n_anomaly += 1
            else:
                cv2.imwrite(str(out_normal / fname), small)
                n_normal += 1
        idx += 1

    cap.release()
    return n_normal, n_anomaly


def prepare(every: int = 10):
    if FLAT:
        out_normal  = OUT / 'Normal'
        out_anomaly = OUT / 'Anomaly'
    else:
        out_normal  = OUT / 'train' / 'Normal'
        out_anomaly = OUT / 'train' / 'Anomaly'

    total_normal = total_anomaly = 0

    # ── Training videos (all Normal) ──────────────────────────────────────────
    train_vid_paths = sorted(TRAIN_VIDS.glob('*.avi'))
    print(f'Processing {len(train_vid_paths)} training videos (all Normal) ...')
    for vp in train_vid_paths:
        n, _ = extract_video(vp, out_normal, None, None, every, vp.stem)
        total_normal += n
        print(f'  {vp.name}: {n} normal frames')

    # ── Testing videos (labeled Normal + Anomaly) ─────────────────────────────
    test_vid_paths = sorted(TEST_VIDS.glob('*.avi'))
    print(f'\nProcessing {len(test_vid_paths)} testing videos (labeled) ...')
    for vp in test_vid_paths:
        vid_id = int(vp.stem)
        labels = load_frame_labels(vid_id)
        n_norm, n_anom = extract_video(
            vp, out_normal, out_anomaly, labels, every, vp.stem)
        total_normal  += n_norm
        total_anomaly += n_anom
        print(f'  {vp.name}: {n_norm} normal, {n_anom} anomaly')

    print('\n=== Summary ===')
    print(f'  Normal:  {total_normal:5d}')
    print(f'  Anomaly: {total_anomaly:5d}')
    if FLAT:
        print('  (flat layout — image_loader will do random 70/15/15 split)')
    print(f'\nDataset ready at: {OUT}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--every', type=int, default=10,
                        help='Sample 1 in N frames (default 10)')
    args = parser.parse_args()
    prepare(args.every)
