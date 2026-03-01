"""
TemporalSmoother — sliding-window EMA over frame-level predictions.

A single surveillance frame is an unreliable signal. A person bending down
looks anomalous in isolation; in sequence it's just picking something up.
This smoother maintains a rolling buffer of recent frame probabilities and
returns both the raw per-frame prediction and a temporally-smoothed one.

  • window  : number of frames to buffer (default 8 ≈ 0.3s at 25fps)
  • alpha   : EMA weight for the newest frame (default 0.4)

Usage:
    smoother = TemporalSmoother(window=8)
    result   = smoother.update(probs)   # probs: np.ndarray (n_classes,)
    result['smoothed_pred']        # class index after smoothing
    result['smoothed_confidence']  # smoothed softmax score
    result['is_stable']            # True once buffer is half-full
    result['anomaly_streak']       # consecutive anomaly frames in buffer
"""

import numpy as np
from collections import deque


class TemporalSmoother:
    """
    Per-stream EMA smoother over class probability vectors.

    Parameters
    ----------
    window : int   — rolling buffer length in frames
    alpha  : float — EMA weight given to the newest frame (0 < alpha ≤ 1)
    anomaly_class : int — class index treated as 'anomaly' for streak counting
    """

    def __init__(self, window: int = 8, alpha: float = 0.4, anomaly_class: int = 1):
        self.window        = window
        self.alpha         = alpha
        self.anomaly_class = anomaly_class
        self._buffer: deque = deque(maxlen=window)

    def update(self, probs: np.ndarray) -> dict:
        """
        Accept frame-level softmax probabilities, return smoothed stats.

        Parameters
        ----------
        probs : np.ndarray, shape (n_classes,)

        Returns
        -------
        dict with keys:
          smoothed_probs       list[float]
          smoothed_pred        int
          smoothed_confidence  float
          window_size          int   (frames buffered so far)
          is_stable            bool  (True once window is half-full)
          anomaly_streak       int   (consecutive anomaly frames at tail of buffer)
          frame_pred           int   (raw single-frame prediction, unsmoothed)
          frame_confidence     float
        """
        probs = np.asarray(probs, dtype=np.float32)
        self._buffer.append(probs)

        # EMA: weight newest frame by alpha, older frames decay
        smoothed = self._buffer[0].copy()
        for p in list(self._buffer)[1:]:
            smoothed = self.alpha * p + (1.0 - self.alpha) * smoothed

        smoothed_pred  = int(np.argmax(smoothed))
        frame_pred     = int(np.argmax(probs))

        # Count how many of the last frames are anomaly (streak from most recent)
        buf_list = list(self._buffer)
        streak = 0
        for p in reversed(buf_list):
            if int(np.argmax(p)) == self.anomaly_class:
                streak += 1
            else:
                break

        return {
            'smoothed_probs':      smoothed.tolist(),
            'smoothed_pred':       smoothed_pred,
            'smoothed_confidence': round(float(smoothed[smoothed_pred]), 4),
            'window_size':         len(self._buffer),
            'is_stable':           len(self._buffer) >= max(1, self.window // 2),
            'anomaly_streak':      streak,
            'frame_pred':          frame_pred,
            'frame_confidence':    round(float(probs[frame_pred]), 4),
        }

    def reset(self):
        """Clear the buffer (call when switching to a new camera stream)."""
        self._buffer.clear()

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0
