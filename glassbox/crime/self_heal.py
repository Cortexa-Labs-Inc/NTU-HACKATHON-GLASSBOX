"""
SelfHealingLoop — orchestrates the iterative failure-correction cycle.

Each healing round:
  1. Evaluate model on validation set → collect wrong predictions
  2. K-means cluster failures in feature space → failure mode centroids
  3. Attribute each centroid to dominant feature chunk (via exact decomposition)
  4. Gaussian-perturb around each centroid → synthetic correct-label samples
  5. Retrain on (original training data + all synthetic data accumulated so far)
  6. Track val-AUC improvement; save best model state
  7. Report per-cluster failure attribution for user notification

The loop stops when:
  - No failures remain (fully healed)
  - Val-AUC stops improving for `patience` consecutive rounds
  - max_rounds reached

After the loop, the best model state (highest val-AUC) is restored.
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score

from crime.failure_detector import FailureModeDetector
from crime.perturber import GaussianPerturber


class SelfHealingLoop:
    """
    Parameters
    ----------
    model          : CrimeGlassboxNet
    X_train        : np.ndarray  (N_train, F)  normalised
    y_train        : np.ndarray  (N_train,)
    X_val          : np.ndarray  (N_val, F)    normalised
    y_val          : np.ndarray  (N_val,)
    class_names    : list[str]
    chunk_names    : list[str]
    n_clusters     : int  — K for K-means failure clustering
    n_synthetic    : int  — synthetic samples per failure cluster per round
    sigma_scale    : float — Gaussian perturbation width (fraction of cluster σ)
    lr             : float — learning rate for fine-tuning
    epochs_per_round : int — retraining epochs per healing round
    max_rounds     : int  — maximum healing rounds
    patience       : int  — early-stop patience (rounds without AUC gain)
    lambda_gate    : float — ghost gate L1 regularisation weight during healing
    """

    def __init__(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        class_names:     list = None,
        chunk_names:     list = None,
        n_clusters:      int   = 5,
        n_synthetic:     int   = 50,
        sigma_scale:     float = 0.3,
        lr:              float = 1e-3,
        epochs_per_round: int  = 30,
        max_rounds:      int   = 5,
        patience:        int   = 2,
        lambda_gate:     float = 0.004,
        batch_size:      int   = 32,
    ):
        self.model          = model
        self.X_train_orig   = X_train.copy()
        self.y_train_orig   = y_train.copy()
        self.X_val          = X_val
        self.y_val          = y_val
        self.class_names    = class_names or []
        self.chunk_names    = chunk_names or []
        self.n_clusters     = n_clusters
        self.n_synthetic    = n_synthetic
        self.sigma_scale    = sigma_scale
        self.lr             = lr
        self.epochs_per_round = epochs_per_round
        self.max_rounds     = max_rounds
        self.patience       = patience
        self.lambda_gate    = lambda_gate
        self.batch_size     = batch_size

        self.perturber       = GaussianPerturber(sigma_scale=sigma_scale)
        self.history         = []      # list of round records
        self.best_model_state = None
        self.best_val_auc    = 0.0

        # Accumulated synthetic data (grows across rounds)
        self._synth_X: list = []
        self._synth_y: list = []

        # Latest failure detector (for API query without re-running)
        self.last_detector: FailureModeDetector | None = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _eval_auc(self, X: np.ndarray, y: np.ndarray) -> float:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(X))
            probs  = torch.softmax(logits, dim=1).numpy()
        n_c = probs.shape[1]
        if n_c == 2:
            return roc_auc_score(y, probs[:, 1])
        try:
            return roc_auc_score(y, probs, multi_class='ovr')
        except ValueError:
            return float(np.mean(probs.argmax(axis=1) == y))

    def _retrain(self, X_aug: np.ndarray, y_aug: np.ndarray):
        dataset   = TensorDataset(torch.FloatTensor(X_aug), torch.LongTensor(y_aug))
        loader    = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(self.epochs_per_round):
            for X_b, y_b in loader:
                optimizer.zero_grad()
                logits = self.model(X_b)
                loss   = criterion(logits, y_b)
                if hasattr(self.model, 'get_gate_l1_loss'):
                    loss = loss + self.lambda_gate * self.model.get_gate_l1_loss()
                loss.backward()
                optimizer.step()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> list:
        """
        Execute the full self-healing loop.

        Returns
        -------
        history : list of round records, each containing:
            round, val_auc_before, val_auc_after, n_failures, n_synthetic,
            clusters (with chunk attribution), perturbation_report, status
        """
        no_improve = 0

        for round_num in range(1, self.max_rounds + 1):
            t0 = time.time()

            # ── Eval before this round ─────────────────────────────────────
            val_auc_before = self._eval_auc(self.X_val, self.y_val)
            self.model.eval()
            with torch.no_grad():
                preds = self.model(torch.FloatTensor(self.X_val)).argmax(dim=1).numpy()
            val_acc_before = float(np.mean(preds == self.y_val))

            # ── Collect failures ───────────────────────────────────────────
            detector = FailureModeDetector(
                self.model,
                class_names=self.class_names,
                chunk_names=self.chunk_names,
            )
            X_fail, y_fail, y_pred_fail = detector.collect_failures(
                self.X_val, self.y_val
            )
            self.last_detector = detector

            record = {
                'round':          round_num,
                'val_auc_before': round(val_auc_before, 4),
                'val_acc_before': round(val_acc_before, 4),
                'n_failures':     int(len(X_fail)),
                'clusters':       [],
                'perturbation_report': [],
                'n_synthetic':    0,
                'val_auc_after':  round(val_auc_before, 4),
                'val_acc_after':  round(val_acc_before, 4),
                'auc_delta':      0.0,
                'elapsed_s':      0.0,
                'status':         'ok',
            }

            if len(X_fail) == 0:
                record['status'] = 'healed'
                self.history.append(record)
                print(f"  Round {round_num}: HEALED — no failures on val set.")
                break

            # ── Cluster failures ───────────────────────────────────────────
            detector.fit(X_fail, y_fail, y_pred_fail, n_clusters=self.n_clusters)
            failure_report = detector.get_failure_report()

            # Enrich clusters with chunk attribution
            for c in failure_report['clusters']:
                k    = c['cluster_id']
                attr = detector.attribute_cluster(k)
                c['chunk_attribution'] = attr.get('blame_scores', {}) if attr else {}
            record['clusters'] = failure_report['clusters']

            # ── Generate synthetic data ────────────────────────────────────
            X_s, y_s, perturb_report = self.perturber.perturb_all_clusters(
                detector.cluster_stats,
                n_synthetic_per_cluster=self.n_synthetic,
            )
            record['perturbation_report'] = perturb_report
            record['n_synthetic']         = int(len(y_s))

            if len(y_s) > 0:
                self._synth_X.append(X_s)
                self._synth_y.append(y_s)

            # ── Retrain on original + all synthetic ────────────────────────
            X_aug = np.vstack([self.X_train_orig] + self._synth_X)
            y_aug = np.concatenate([self.y_train_orig] + self._synth_y)
            self._retrain(X_aug, y_aug)

            # ── Eval after retraining ──────────────────────────────────────
            val_auc_after = self._eval_auc(self.X_val, self.y_val)
            self.model.eval()
            with torch.no_grad():
                preds_after = self.model(torch.FloatTensor(self.X_val)).argmax(dim=1).numpy()
            val_acc_after = float(np.mean(preds_after == self.y_val))

            record['val_auc_after'] = round(val_auc_after, 4)
            record['val_acc_after'] = round(val_acc_after, 4)
            record['auc_delta']     = round(val_auc_after - val_auc_before, 4)
            record['elapsed_s']     = round(time.time() - t0, 2)

            print(
                f"  Round {round_num} | failures={len(X_fail)} "
                f"| synth={len(y_s)} | AUC {val_auc_before:.4f}→{val_auc_after:.4f} "
                f"(Δ{record['auc_delta']:+.4f}) | {record['elapsed_s']}s"
            )

            # Best model tracking
            if val_auc_after > self.best_val_auc:
                self.best_val_auc  = val_auc_after
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            self.history.append(record)

            if no_improve >= self.patience:
                record['status'] = 'early_stop'
                print(f"  Early stop: no AUC gain for {self.patience} rounds.")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.history

    # ── Runtime query: is this sample near a failure cluster? ─────────────────

    def get_proximity_warning(
        self, x: np.ndarray, distance_threshold: float = 3.0
    ) -> dict:
        """
        Given a single (normalised) feature vector, check proximity to known
        failure centroids.  Returns a user-facing warning dict.

        Parameters
        ----------
        x                  : (F,) normalised feature vector
        distance_threshold : warn if nearest centroid is within this distance
        """
        if self.last_detector is None or not self.last_detector.cluster_stats:
            return {'status': 'no_failure_map', 'warning': False}

        idx, dist = self.last_detector.nearest_cluster(x)
        if idx < 0:
            return {'status': 'no_failure_map', 'warning': False}

        stats = self.last_detector.cluster_stats[idx]
        attr  = self.last_detector.attribute_cluster(idx)

        if dist > distance_threshold:
            return {
                'status':  'safe',
                'warning': False,
                'nearest_cluster': idx,
                'distance':        round(dist, 4),
            }

        return {
            'status':          'warning',
            'warning':         True,
            'nearest_cluster': idx,
            'distance':        round(dist, 4),
            'cluster_info':    stats,
            'dominant_chunk':  attr['dominant_chunk'] if attr else 'Unknown',
            'blame_scores':    attr.get('blame_scores', {}) if attr else {},
            'message': (
                f"Sample is near Failure Cluster #{idx} "
                f"(distance={dist:.2f}, {stats['n_samples']} historical failures). "
                f"Dominant failure chunk: {attr['dominant_chunk'] if attr else 'Unknown'}. "
                f"Predicted '{stats['pred_name']}' when truth was '{stats['true_name']}'. "
                f"Perturbation recipe: generate '{stats['true_name']}' samples "
                f"with σ_scale={self.sigma_scale}."
            ),
        }

    def get_summary(self) -> dict:
        """Compact summary of the healing run for the API."""
        if not self.history:
            return {'status': 'not_run'}

        last = self.history[-1]
        first = self.history[0]
        return {
            'rounds_run':     len(self.history),
            'best_val_auc':   round(self.best_val_auc, 4),
            'final_failures': last['n_failures'],
            'total_synthetic': sum(r['n_synthetic'] for r in self.history),
            'auc_start':      first['val_auc_before'],
            'auc_end':        last['val_auc_after'],
            'auc_total_gain': round(last['val_auc_after'] - first['val_auc_before'], 4),
            'status':         last.get('status', 'ok'),
            'history':        self.history,
        }
