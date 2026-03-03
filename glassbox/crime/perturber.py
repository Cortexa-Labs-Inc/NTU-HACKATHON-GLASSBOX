"""
GaussianPerturber — generates synthetic corrective training data.

For each failure mode cluster:
  centroid  = K-means center of failure samples in normalised feature space
  σ         = intra-cluster std per feature dimension × sigma_scale
  x_synth   ~ N(centroid, diag(σ²))
  y_synth   = true_class  (the label the model should have predicted)

The synthetic data teaches the model to correctly classify samples near each
failure centroid, healing that failure mode in subsequent training rounds.

The perturbation recipe (μ, σ per cluster) is also exposed as a human-readable
data-generation instruction:
  "Generate 50 'Robbery' samples with μ=0.24, σ=0.06 in normalised feature space"
"""

import numpy as np


class GaussianPerturber:
    """
    Parameters
    ----------
    sigma_scale : float
        Scale factor applied to intra-cluster σ.
        0.3 = perturb within 30% of cluster spread (tight around centroid).
        1.0 = perturb at the natural cluster spread.
    clip_range  : (float, float)
        Clip generated values to this range in normalised space.
        Default (−5, 5) is ≈5 standard deviations, very conservative.
    """

    def __init__(self, sigma_scale: float = 0.3, clip_range: tuple = (-5.0, 5.0)):
        self.sigma_scale = sigma_scale
        self.clip_range  = clip_range

    # ── Single cluster ────────────────────────────────────────────────────────

    def perturb_cluster(
        self,
        centroid: np.ndarray,
        sigma: np.ndarray,
        correct_label: int,
        n_synthetic: int = 50,
        seed: int = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples around one failure centroid.

        Parameters
        ----------
        centroid      : (F,) feature vector (normalised space)
        sigma         : (F,) per-feature std of this cluster
        correct_label : int — the ground-truth label for synthetic samples
        n_synthetic   : int — number of samples to generate
        seed          : optional RNG seed for reproducibility

        Returns
        -------
        X_synth : (n_synthetic, F)
        y_synth : (n_synthetic,)  all = correct_label
        """
        rng = np.random.default_rng(seed)
        eff_sigma = np.maximum(sigma * self.sigma_scale, 0.01)

        noise   = rng.standard_normal((n_synthetic, len(centroid)))
        X_synth = centroid + noise * eff_sigma
        X_synth = np.clip(X_synth, self.clip_range[0], self.clip_range[1])
        y_synth = np.full(n_synthetic, correct_label, dtype=int)

        return X_synth.astype(np.float32), y_synth

    # ── All clusters in one call ──────────────────────────────────────────────

    def perturb_all_clusters(
        self,
        cluster_stats: dict,
        n_synthetic_per_cluster: int = 50,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Generate synthetic data for every failure cluster.

        Parameters
        ----------
        cluster_stats : dict from FailureModeDetector.cluster_stats
            {k: {'centroid': ..., 'sigma': ..., 'true_class': int, ...}}
        n_synthetic_per_cluster : int

        Returns
        -------
        X_all   : (total_synthetic, F)  or empty array
        y_all   : (total_synthetic,)    or empty array
        reports : list[dict] — per-cluster synthesis report
        """
        all_X, all_y = [], []
        reports      = []

        for k, stats in cluster_stats.items():
            centroid = stats['centroid']
            sigma    = stats['sigma']
            label    = stats['true_class']

            X_s, y_s = self.perturb_cluster(
                centroid, sigma, label,
                n_synthetic=n_synthetic_per_cluster, seed=k,
            )
            all_X.append(X_s)
            all_y.append(y_s)

            reports.append({
                'cluster_id':      k,
                'correct_label':   label,
                'correct_name':    stats.get('true_name', f'Class{label}'),
                'n_generated':     n_synthetic_per_cluster,
                'centroid_norm':   round(float(np.linalg.norm(centroid)), 4),
                'sigma_mean':      round(float(sigma.mean()), 4),
                'sigma_max':       round(float(sigma.max()), 4),
                'effective_sigma': round(float((sigma * self.sigma_scale).mean()), 4),
            })

        if all_X:
            return np.vstack(all_X), np.concatenate(all_y), reports
        return np.empty((0,)), np.empty((0,), dtype=int), []

    # ── Human-readable recipe ─────────────────────────────────────────────────

    def get_perturbation_recipe(self, cluster_stats: dict, class_names: list = None) -> list:
        """
        Return human-readable data-generation instructions per cluster.

        Example output:
          {
            'cluster_id': 2,
            'instruction': "Generate 'Robbery' samples around centroid ...",
            'mu_summary':  "norm=1.23, mean=0.04",
            'sigma_summary': "mean=0.08, max=0.31, effective(×0.3)=0.024",
          }
        """
        def cname(idx):
            if class_names and 0 <= idx < len(class_names):
                return class_names[idx]
            return f'Class{idx}'

        recipes = []
        for k, stats in cluster_stats.items():
            centroid = stats['centroid']
            sigma    = stats['sigma']
            label    = stats['true_class']
            eff      = sigma * self.sigma_scale

            recipes.append({
                'cluster_id':   k,
                'n_failures':   stats['n_samples'],
                'correct_label': label,
                'correct_name':  cname(label),
                'instruction': (
                    f"Cluster {k} ({stats['n_samples']} failures → "
                    f"predicted '{stats.get('pred_name', '?')}' "
                    f"but truth is '{stats.get('true_name', cname(label))}'): "
                    f"generate {cname(label)} samples via "
                    f"N(centroid, σ×{self.sigma_scale})"
                ),
                'mu_summary':    (
                    f"centroid norm={np.linalg.norm(centroid):.3f}, "
                    f"mean={centroid.mean():.3f}"
                ),
                'sigma_summary': (
                    f"σ_mean={sigma.mean():.4f}, σ_max={sigma.max():.4f}, "
                    f"effective σ_mean={eff.mean():.4f}"
                ),
            })
        return recipes
