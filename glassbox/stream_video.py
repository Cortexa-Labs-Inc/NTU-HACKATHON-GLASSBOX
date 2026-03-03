#!/usr/bin/env python3
"""
stream_video.py — Simulate a live CCTV feed by streaming extracted frames to the API.

Usage:
  python3 stream_video.py                  # stream all videos at 8 fps
  python3 stream_video.py --video 01       # stream only video 01
  python3 stream_video.py --fps 15         # faster playback
  python3 stream_video.py --api http://localhost:8001 --video 03 --fps 10

Frame naming convention: CUHK_Avenue/{Normal,Anomaly}/{vid}_f{frame:06d}.png
The dashboard alert log updates in real time as anomalies are detected.
"""

import argparse
import glob
import os
import sys
import time
import requests

# ── ANSI colours ──────────────────────────────────────────────────────────────
RED    = '\033[91m'
GREEN  = '\033[92m'
AMBER  = '\033[93m'
INDIGO = '\033[95m'
MUTED  = '\033[90m'
BOLD   = '\033[1m'
RESET  = '\033[0m'


def collect_frames(data_dir, video_id=None):
    """Return list of (path, true_label, vid, frame_num) sorted chronologically."""
    frames = []
    for label, cls in [('Normal', 0), ('Anomaly', 1)]:
        for path in glob.glob(os.path.join(data_dir, label, '*.png')):
            fname = os.path.basename(path)
            parts = fname.split('_f')
            if len(parts) != 2:
                continue
            vid = parts[0]
            if video_id and vid != video_id:
                continue
            try:
                fnum = int(parts[1].replace('.png', ''))
            except ValueError:
                continue
            frames.append((path, cls, vid, fnum))

    frames.sort(key=lambda x: (x[2], x[3]))
    return frames


def stream(frames, fps, api_url):
    delay       = 1.0 / fps
    current_vid = None
    n_alerts    = 0
    n_correct   = 0

    print(f"\n{BOLD}CrimeVisionGlassbox — Live Stream{RESET}")
    print(f"  Frames : {len(frames)}")
    print(f"  FPS    : {fps}  ({delay:.3f}s/frame)")
    print(f"  API    : {api_url}")
    print(f"  Dashboard alert log refreshes every 5 s automatically.\n")

    for i, (path, true_cls, vid, fnum) in enumerate(frames):

        # Reset temporal state when switching between videos
        if vid != current_vid:
            if current_vid is not None:
                try:
                    requests.post(f'{api_url}/reset_temporal', timeout=2)
                except Exception:
                    pass
                print()
            print(f"{MUTED}── Video {vid} {'─'*44}{RESET}")
            current_vid = vid

        t0 = time.time()

        try:
            with open(path, 'rb') as f:
                resp = requests.post(
                    f'{api_url}/predict',
                    files={'file': (os.path.basename(path), f, 'image/png')},
                    timeout=10,
                )
            resp.raise_for_status()
            d = resp.json()
        except Exception as e:
            print(f"{MUTED}[{i+1:4d}] f{fnum:06d}  ERROR: {e}{RESET}")
            time.sleep(delay)
            continue

        pred      = d.get('predicted_class', '?')
        conf      = d.get('confidence', 0)
        alert     = d.get('alert_triggered', False)
        temporal  = d.get('temporal', {})
        streak    = temporal.get('anomaly_streak', 0)
        lstm      = d.get('lstm_temporal', {})
        contribs  = d.get('chunk_contributions', {})
        true_name = 'Anomaly' if true_cls == 1 else 'Normal'

        is_anomaly = 'anomaly' in pred.lower()
        correct    = is_anomaly == (true_cls == 1)
        if correct:
            n_correct += 1
        if alert:
            n_alerts += 1

        # Dominant chunk by absolute logit contribution
        dominant = ''
        if contribs:
            dominant = max(
                contribs,
                key=lambda k: abs(contribs[k].get('pred_push',
                                  contribs[k].get('disease_push', 0)))
            )

        pred_col   = RED if is_anomaly else GREEN
        mark_col   = GREEN if correct else RED
        mark       = '✓' if correct else '✗'

        streak_str = (f" {RED}[STREAK {streak}]{RESET}" if streak >= 3
                      else f" {AMBER}[streak {streak}]{RESET}" if streak > 0
                      else '')
        alert_str  = f"  {RED}{BOLD}⚑ ALERT{RESET}" if alert else ''

        lstm_str = ''
        if lstm.get('lstm_pred'):
            lc = RED if 'anomaly' in lstm['lstm_pred'].lower() else GREEN
            lstm_str = f"  LSTM:{lc}{lstm['lstm_pred'][:3]}{RESET}({lstm['lstm_confidence']*100:.0f}%)"

        chunk_str = f"  {INDIGO}{dominant}{RESET}" if dominant else ''

        print(
            f"{MUTED}[{i+1:4d}/{len(frames)}] f{fnum:06d}{RESET}"
            f"  {pred_col}{pred[:7]:7s}{RESET} {conf*100:4.0f}%"
            f"  true:{MUTED}{true_name[:3]}{RESET}"
            f"  {mark_col}{mark}{RESET}"
            f"{lstm_str}{chunk_str}{streak_str}{alert_str}"
        )

        elapsed = time.time() - t0
        remaining = delay - elapsed
        if remaining > 0:
            time.sleep(remaining)

    n = len(frames)
    print(f"\n{BOLD}── Summary {'─'*38}{RESET}")
    print(f"  Frames processed : {n}")
    print(f"  Correct          : {n_correct}/{n}  ({n_correct/n*100:.1f}%)")
    print(f"  Alerts fired     : {n_alerts}")
    print(f"  Full alert log   : {api_url}/alerts\n")


def main():
    parser = argparse.ArgumentParser(
        description='Stream CUHK Avenue frames to CrimeVisionGlassbox API.'
    )
    parser.add_argument('--video', default=None,
                        help='Video ID to stream, e.g. "01" (default: all)')
    parser.add_argument('--fps',   type=float, default=8.0,
                        help='Playback FPS (default: 8)')
    parser.add_argument('--api',   default='http://localhost:8001',
                        help='API base URL')
    parser.add_argument('--data',  default=None,
                        help='Path to CUHK_Avenue dir (auto-detected by default)')
    args = parser.parse_args()

    # Auto-detect data dir relative to this script
    data_dir = args.data or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CUHK_Avenue')
    if not os.path.isdir(data_dir):
        print(f"{RED}CUHK_Avenue directory not found at {data_dir}{RESET}")
        print("Run prepare_cuhk.py first, or pass --data /path/to/CUHK_Avenue")
        sys.exit(1)

    # Health check
    try:
        r = requests.get(f'{args.api}/health', timeout=3)
        r.raise_for_status()
        h = r.json()
        print(f"{GREEN}API online{RESET}  model={h.get('model_loaded')}  clusters={h.get('n_clusters')}")
    except Exception as e:
        print(f"{RED}API offline at {args.api}{RESET}: {e}")
        print("Start it with:  python3 -m uvicorn api.crime_app:app --reload --port 8001")
        sys.exit(1)

    frames = collect_frames(data_dir, video_id=args.video)
    if not frames:
        target = f"video '{args.video}'" if args.video else "any video"
        print(f"No frames found for {target} in {data_dir}")
        sys.exit(1)

    stream(frames, fps=args.fps, api_url=args.api)


if __name__ == '__main__':
    main()
