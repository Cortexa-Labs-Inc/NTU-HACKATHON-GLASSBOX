"""
TemporalLSTMHead — learned sequence classifier over chunk embeddings.

Architecture
------------
Input: sequence of CNN feature vectors (T, n_features)  e.g. (8, 128)
  ↓
LSTM(input=n_features, hidden=64, layers=2, dropout=0.2)
  ↓  hidden state of last step
Linear(64 → n_classes)
  ↓
Logits → softmax → prediction

Training
--------
  Sequences are constructed from consecutive frames within the same video
  (parsed from filename pattern {vid_id}_f{frame_num:06d}.png).
  Labels: majority class across the sequence window.
  CNN + Glassbox weights stay frozen — only the LSTM head is trained.

Usage
-----
  # Build and train
  head = TemporalLSTMHead(n_features=128, n_classes=2)
  head.fit(sequences, labels, epochs=20)

  # Inference (stateful — maintains hidden state across frames)
  head.reset_state()
  for frame_feat in stream:
      pred, conf = head.step(frame_feat)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict


# ── Model ─────────────────────────────────────────────────────────────────────

class TemporalLSTMHead(nn.Module):
    """
    Small LSTM classifier operating on a sequence of frame-level feature vectors.

    Parameters
    ----------
    n_features : input feature dimension (CNN output, default 128)
    hidden_dim : LSTM hidden size (default 64)
    n_layers   : number of LSTM layers (default 2)
    n_classes  : output classes (default 2)
    dropout    : dropout between LSTM layers (default 0.2)
    """

    def __init__(self, n_features: int = 128, hidden_dim: int = 64,
                 n_layers: int = 2, n_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.n_classes  = n_classes

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, n_classes)

        # Stateful inference state (h, c) — reset between streams
        self._h: torch.Tensor | None = None
        self._c: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, n_features)
        Returns logits: (batch, n_classes)
        """
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # use last hidden state

    # ── Stateful streaming inference ──────────────────────────────────────────

    def reset_state(self):
        self._h = None
        self._c = None

    def step(self, feat_vec: np.ndarray) -> tuple[int, float]:
        """
        Process a single new frame feature vector, updating internal state.
        Returns (predicted_class_idx, confidence).

        feat_vec : np.ndarray (n_features,)
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(feat_vec).unsqueeze(0).unsqueeze(0)  # (1,1,F)
            if self._h is None:
                out, (self._h, self._c) = self.lstm(x)
            else:
                out, (self._h, self._c) = self.lstm(x, (self._h, self._c))
            logits = self.head(out[:, -1, :])
            probs  = F.softmax(logits, dim=-1)[0]
            pred   = int(probs.argmax().item())
            conf   = float(probs[pred].item())
        return pred, conf

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X_seqs: np.ndarray, y_seqs: np.ndarray,
            epochs: int = 25, lr: float = 1e-3,
            val_X: np.ndarray = None, val_y: np.ndarray = None,
            verbose: bool = True) -> list:
        """
        Train on pre-extracted sequences.

        X_seqs : (N, T, n_features)
        y_seqs : (N,)  integer class labels
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        X_t = torch.FloatTensor(X_seqs)
        y_t = torch.LongTensor(y_seqs)

        history = []
        for epoch in range(1, epochs + 1):
            self.train()
            # Shuffle
            idx = torch.randperm(len(X_t))
            X_t, y_t = X_t[idx], y_t[idx]
            # Mini-batch
            bs, total_loss = 32, 0.0
            for i in range(0, len(X_t), bs):
                xb, yb = X_t[i:i+bs], y_t[i:i+bs]
                optimizer.zero_grad()
                loss = criterion(self(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            rec = {'epoch': epoch, 'train_loss': round(total_loss, 4)}

            if val_X is not None and val_y is not None:
                self.eval()
                with torch.no_grad():
                    val_logits = self(torch.FloatTensor(val_X))
                    val_preds  = val_logits.argmax(dim=1).numpy()
                    val_acc    = float((val_preds == val_y).mean())
                    rec['val_acc'] = round(val_acc, 4)

            history.append(rec)
            if verbose and epoch % 5 == 0:
                msg = f"  LSTM epoch {epoch:3d} | loss={rec['train_loss']:.4f}"
                if 'val_acc' in rec:
                    msg += f" | val_acc={rec['val_acc']:.4f}"
                print(msg)

        return history


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_sequence_dataset(
    data_dir: str,
    class_names: list,
    model,
    transform,
    seq_len: int = 8,
    stride:  int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sequence dataset from frame files named {vid_id}_f{frame_num:06d}.png.

    Steps:
      1. Parse all filenames → (video_id, frame_num, class_idx, path)
      2. Group by video_id, sort by frame_num within each video
      3. Extract CNN features for each frame (using model.extract)
      4. Slide a window of seq_len with stride over each video's features
      5. Label each sequence by majority class in the window

    Returns (X_seqs, y_seqs):
      X_seqs : (N, seq_len, n_features)
      y_seqs : (N,) int
    """
    data_path = Path(data_dir)

    # Collect all frames with sequence metadata
    video_frames = defaultdict(list)   # vid_id → [(frame_num, feat_path, label_idx)]

    model.eval()
    print("[TemporalLSTM] Extracting features in sequence order ...")

    all_files = []
    for label_idx, label_name in enumerate(class_names):
        class_dir = data_path / label_name
        if not class_dir.exists():
            continue
        for img_path in sorted(class_dir.glob('*.png')):
            name = img_path.stem   # e.g. "01_f000080"
            parts = name.split('_f')
            if len(parts) == 2:
                vid_id    = parts[0]
                frame_num = int(parts[1])
            else:
                vid_id    = 'unk'
                frame_num = len(all_files)
            all_files.append((vid_id, frame_num, label_idx, img_path))

    # Extract features
    from PIL import Image
    feat_map = {}   # path → feat np.ndarray
    with torch.no_grad():
        for _, _, _, img_path in all_files:
            img    = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0)
            feat   = model.extract(tensor).numpy().flatten()
            feat_map[img_path] = feat

    # Group by video
    for vid_id, frame_num, label_idx, img_path in all_files:
        video_frames[vid_id].append((frame_num, label_idx, feat_map[img_path]))

    # Build sequences
    X_seqs, y_seqs = [], []
    for vid_id, frames in video_frames.items():
        frames_sorted = sorted(frames, key=lambda f: f[0])
        feats  = np.array([f[2] for f in frames_sorted])
        labels = np.array([f[1] for f in frames_sorted])

        if len(feats) < seq_len:
            continue

        for start in range(0, len(feats) - seq_len + 1, stride):
            seq   = feats[start:start + seq_len]
            label = int(np.bincount(labels[start:start + seq_len]).argmax())
            X_seqs.append(seq)
            y_seqs.append(label)

    X_seqs = np.array(X_seqs, dtype=np.float32)
    y_seqs = np.array(y_seqs, dtype=np.int64)
    print(f"[TemporalLSTM] Built {len(X_seqs)} sequences "
          f"(seq_len={seq_len}, stride={stride}) from {len(video_frames)} videos")
    return X_seqs, y_seqs
