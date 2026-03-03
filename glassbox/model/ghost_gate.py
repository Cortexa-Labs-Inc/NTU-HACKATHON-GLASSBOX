import torch
import torch.nn as nn
import torch.nn.functional as F


class GhostSignalGate(nn.Module):
    """
    Per-sample Ghost Signal Gate.

    Original design used a single global scalar α shared across all patients.
    This version computes α per patient:

        α(x) = sigmoid(gate_net(concat(chunk_a, chunk_b)))

    where gate_net is a single linear layer initialized so that:
        - bias = -3 → gates start nearly closed (sigmoid(-3) ≈ 0.047)
        - weights ≈ 0 → initial output ≈ -3 for all inputs

    This means:
        - A gate only opens for a specific patient if that patient's chunk
          embeddings push the gate_net output above the softplus penalty.
        - α now varies per patient, enabling genuine per-patient z-score
          anomaly detection ("this gate was unusually active for YOU").
        - gate_net.bias reflects the population-average gate openness.

    L1 regularisation is done by the parent model via get_gate_l1_loss(),
    which calls softplus(self._last_logit).mean() using the logit stored
    during the most recent forward pass.
    """

    def __init__(self, chunk_a_dim: int, chunk_b_dim: int, gate_name: str):
        super().__init__()
        self.gate_name   = gate_name
        self.chunk_a_dim = chunk_a_dim
        self.chunk_b_dim = chunk_b_dim

        # Ghost attention: combines both chunk outputs → ghost signal
        self.attention = nn.Linear(chunk_a_dim + chunk_b_dim, chunk_b_dim)

        # Per-sample gate: single linear layer → scalar logit per patient
        # bias=-3: gate starts nearly closed; weights~0: no input dependence at init
        self.gate_net = nn.Linear(chunk_a_dim + chunk_b_dim, 1)
        nn.init.zeros_(self.gate_net.weight)
        nn.init.constant_(self.gate_net.bias, -3.0)

        # Stores raw logit from last forward — used by parent model's L1 loss
        self._last_logit: torch.Tensor = None

    def forward(self, chunk_a_out, chunk_b_out):
        """
        Args:
            chunk_a_out: (N, chunk_a_dim)
            chunk_b_out: (N, chunk_b_dim)

        Returns:
            output:          (N, chunk_a_dim)  — gated combination
            alpha_mean:      float             — mean gate weight this batch (for display)
            ghost_magnitude: float             — mean magnitude of ghost contribution
        """
        combined = torch.cat([chunk_a_out, chunk_b_out], dim=-1)   # (N, a+b)

        # Ghost signal: attention-weighted chunk_b
        ghost = torch.sigmoid(self.attention(combined)) * chunk_b_out  # (N, b)

        # Per-sample gate weight
        alpha_logit = self.gate_net(combined)               # (N, 1)
        alpha       = torch.sigmoid(alpha_logit)            # (N, 1) ∈ (0,1)

        # Residual addition — exact so audit decomposition holds
        output = chunk_a_out + alpha * ghost                # (N, a)

        # Store raw logit for L1 loss (parent model reads this)
        self._last_logit = alpha_logit                      # (N, 1)

        # Scalars for downstream display / audit
        alpha_mean      = alpha.mean().item()
        ghost_magnitude = (alpha * ghost).norm(dim=-1).mean().item()

        return output, alpha_mean, ghost_magnitude

    def get_gate_weight(self) -> float:
        """Return baseline gate weight (bias only, input-independent)."""
        return torch.sigmoid(self.gate_net.bias[0]).item()

    def get_alpha_for_input(self, chunk_a_out: torch.Tensor,
                            chunk_b_out: torch.Tensor) -> torch.Tensor:
        """Return per-sample α values (N,) without running full forward."""
        with torch.no_grad():
            combined = torch.cat([chunk_a_out, chunk_b_out], dim=-1)
            return torch.sigmoid(self.gate_net(combined)).squeeze(-1)
