"""
FailureModeDetector — clusters wrong predictions in feature space.

For each failure:
  • Collect raw (normalised) feature vector + true label + predicted label
  • K-means cluster in feature space  → failure mode centroids
  • For each centroid: run through model → chunk blame attribution
  • Report which cluster + which chunk + what σ to use for perturbation

The feature-space centroid is the 'failure mode' that the Gaussian perturber
will perturb around to generate synthetic corrective training data.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score


class FailureModeDetector:
    """
    Detects and characterises model failure modes.

    Parameters
    ----------
    model       : CrimeGlassboxNet
    class_names : list[str]  — class name strings
    chunk_names : list[str]  — chunk name strings
    """

    def __init__(self, model, class_names: list = None, chunk_names: list = None):
        self.model       = model
        self.class_names = class_names or []
        self.chunk_names = chunk_names or []

        # Populated by fit()
        self.centroids     = None   # (K, F) in feature space
        self.cluster_stats = {}     # {k: {...}}
        self.n_clusters_   = 0

    # ── Step 1: collect failures ──────────────────────────────────────────────

    def collect_failures(
        self, X: np.ndarray, y_true: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run model on X, return (X_fail, y_true_fail, y_pred_fail).
        X : (N, F) normalised feature array
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(X))
            y_pred = logits.argmax(dim=1).numpy()

        mask = y_pred != y_true
        return X[mask], y_true[mask], y_pred[mask]

    # ── Step 2: cluster in feature space ─────────────────────────────────────

    def fit(
        self,
        X_fail: np.ndarray,
        y_fail: np.ndarray,
        y_pred_fail: np.ndarray,
        n_clusters: int = None,
    ) -> 'FailureModeDetector':
        """
        Cluster failure samples. Populates self.centroids and self.cluster_stats.
        """
        n = len(X_fail)
        if n == 0:
            self.centroids = np.empty((0, X_fail.shape[1] if X_fail.ndim > 1 else 0))
            self.n_clusters_ = 0
            return self

        n_clusters = n_clusters or min(5, max(2, n // max(3, n // 10)))

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X_fail)

        self.centroids    = km.cluster_centers_   # (K, F)
        self.n_clusters_  = n_clusters

        for k in range(n_clusters):
            mask_k       = labels == k
            cluster_X    = X_fail[mask_k]
            cluster_y    = y_fail[mask_k]
            cluster_pred = y_pred_fail[mask_k]

            # Dominant error class pair
            pairs = list(zip(cluster_y.tolist(), cluster_pred.tolist()))
            if pairs:
                dominant_pair = max(set(pairs), key=pairs.count)
            else:
                dominant_pair = (0, 0)

            true_c, pred_c = dominant_pair
            sigma = cluster_X.std(axis=0)
            sigma = np.maximum(sigma, 1e-3)   # floor to avoid zero σ

            self.cluster_stats[k] = {
                'n_samples':     int(mask_k.sum()),
                'centroid':      self.centroids[k],
                'sigma':         sigma,
                'true_class':    int(true_c),
                'pred_class':    int(pred_c),
                'true_name':     self._cn(true_c),
                'pred_name':     self._cn(pred_c),
                'error_type': (
                    'false_positive' if pred_c != 0 and true_c == 0
                    else 'false_negative' if pred_c == 0 and true_c != 0
                    else 'misclassification'
                ),
            }

        return self

    # ── Step 3: chunk attribution per centroid ────────────────────────────────

    def attribute_cluster(self, cluster_idx: int) -> dict | None:
        """
        Run the failure centroid through the model → chunk blame attribution.
        Returns None if no centroids or model lacks get_class_pair_contributions.
        """
        if self.centroids is None or len(self.centroids) == 0:
            return None

        stats    = self.cluster_stats[cluster_idx]
        centroid = self.centroids[cluster_idx]
        x_tensor = torch.FloatTensor(centroid).unsqueeze(0)   # (1, F)

        self.model.eval()
        with torch.no_grad():
            try:
                logits, audit = self.model(x_tensor, return_audit=True)
            except TypeError:
                logits = self.model(x_tensor)
                audit  = {}
            pred = int(logits.argmax(dim=1).item())

        true_c = stats['true_class']

        # If the model now correctly classifies this centroid (post-healing),
        # use the most-confused wrong class for blame attribution — otherwise
        # blame scores would all be zero (pred_push == true_push trivially).
        blame_pred = pred
        if pred == true_c and logits.shape[1] > 1:
            probs = torch.softmax(logits, dim=1)[0].clone()
            probs[true_c] = -1.0
            blame_pred = int(probs.argmax().item())

        result = {
            'centroid_pred':      pred,
            'centroid_pred_name': self._cn(pred),
            'true_class':         true_c,
            'true_class_name':    self._cn(true_c),
            'order_decomp':       audit.get('order_decomp', {}),
        }

        if hasattr(self.model, 'get_class_pair_contributions'):
            blame = self.model.get_class_pair_contributions(x_tensor, blame_pred, true_c)
            ranked = sorted(blame.items(), key=lambda kv: abs(kv[1]['blame']), reverse=True)
            result['blame_scores']   = {k: v['blame'] for k, v in blame.items()}
            result['dominant_chunk'] = ranked[0][0] if ranked else 'Unknown'
            result['ranked_chunks']  = [(k, v['blame']) for k, v in ranked]
        else:
            result['dominant_chunk'] = 'Unknown'
            result['blame_scores']   = {}

        return result

    # ── Reports ───────────────────────────────────────────────────────────────

    def get_failure_report(self) -> dict:
        """Full failure cluster report for the API / dashboard."""
        if not self.cluster_stats:
            return {'status': 'no_failures', 'clusters': []}

        clusters = []
        for k, stats in self.cluster_stats.items():
            attr = self.attribute_cluster(k)
            clusters.append({
                'cluster_id':     k,
                'n_samples':      stats['n_samples'],
                'error_type':     stats['error_type'],
                'true_class':     stats['true_class'],
                'pred_class':     stats['pred_class'],
                'true_name':      stats['true_name'],
                'pred_name':      stats['pred_name'],
                'dominant_chunk': attr['dominant_chunk'] if attr else 'Unknown',
                'blame_scores':   attr.get('blame_scores', {}) if attr else {},
                'order_decomp':   attr.get('order_decomp', {}) if attr else {},
                'centroid_norm':  round(float(np.linalg.norm(stats['centroid'])), 4),
                'sigma_mean':     round(float(stats['sigma'].mean()), 4),
            })

        return {
            'status':          'failures_detected',
            'n_clusters':      len(self.cluster_stats),
            'total_failures':  sum(s['n_samples'] for s in self.cluster_stats.values()),
            'clusters':        clusters,
        }

    def nearest_cluster(self, x: np.ndarray) -> tuple[int, float]:
        """
        Find the nearest failure centroid to a feature vector x (1D).
        Returns (cluster_idx, distance).
        """
        if self.centroids is None or len(self.centroids) == 0:
            return -1, float('inf')
        dists = np.linalg.norm(self.centroids - x.reshape(1, -1), axis=1)
        idx   = int(np.argmin(dists))
        return idx, float(dists[idx])

    def _cn(self, idx: int) -> str:
        if 0 <= idx < len(self.class_names):
            return self.class_names[idx]
        return f'Class{idx}'
