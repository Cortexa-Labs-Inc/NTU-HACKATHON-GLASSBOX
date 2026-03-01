"""
Parameterized GlassboxNet — works with any chunk configuration.

Implements the full GhostGate spec:
  • Feature Chunking      — any chunk layout (auto-discovered or domain-driven)
  • Ghost Signal Gates    — learned α gates for all upper-triangle chunk pairs
  • Order Decomposition   — per-chunk β gate separating 1st-order vs nth-order paths
  • Sub-chunk MoE         — K learned sub-experts per named chunk (optional)
  • Structural Audit      — chunk contributions, ghost signals, order weights, sub-routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.chunks import ChunkNet
from model.ghost_gate import GhostSignalGate


class GlassboxNetV2(nn.Module):
    """
    Configurable Glassbox model.

    chunk_sizes:      list of input feature counts per chunk (e.g. [32, 32, 32, 32])
    embed_dim:        shared embedding dimension
    n_classes:        output classes (default 2)
    use_ghost:        if False, bypass ghost gates (ablation mode)
    use_order_decomp: if True, each chunk has 1st-order + nth-order split with β gate
    n_sub_chunks:     if > 1, each named chunk runs a Mixture-of-Sub-Experts with K
                      parallel sub-MLPs and a learned softmax router. The routing
                      weights reveal which mathematical sub-pattern fired per frame.
    """

    def __init__(self, chunk_sizes: list, embed_dim: int = 16,
                 n_classes: int = 2, use_ghost: bool = True,
                 use_order_decomp: bool = False, n_sub_chunks: int = 1):
        super().__init__()
        self.chunk_sizes      = chunk_sizes
        self.embed_dim        = embed_dim
        self.n_chunks         = len(chunk_sizes)
        self.use_ghost        = use_ghost
        self.use_order_decomp = use_order_decomp
        self.n_sub_chunks     = n_sub_chunks

        # ── Chunk subnetworks ──────────────────────────────────────────────────
        self.chunks = nn.ModuleList([
            ChunkNet(sz, [64, 32], embed_dim, f'Chunk{i}',
                     use_order_decomp=use_order_decomp,
                     n_sub_chunks=n_sub_chunks)
            for i, sz in enumerate(chunk_sizes)
        ])

        # ── Ghost Signal gates (all upper-triangle pairs) ──────────────────────
        self.gates = nn.ModuleDict()
        self.gate_pairs = []
        for i in range(self.n_chunks):
            for j in range(i + 1, self.n_chunks):
                name = f'C{i}→C{j}'
                self.gates[name.replace('→', '_')] = GhostSignalGate(
                    embed_dim, embed_dim, name
                )
                self.gate_pairs.append((i, j, name))

        # Classifier
        self.classifier = nn.Linear(embed_dim * self.n_chunks, n_classes)

    def forward(self, x, return_audit=False):
        # ── Chunk forward passes ───────────────────────────────────────────────
        embeddings = []
        all_norms  = {}
        offset = 0
        for i, (chunk, sz) in enumerate(zip(self.chunks, self.chunk_sizes)):
            xi = x[:, offset:offset + sz]
            emb, norms = chunk(xi)
            embeddings.append(emb)
            all_norms.update(norms)
            offset += sz

        # ── Ghost gates ────────────────────────────────────────────────────────
        gated = list(embeddings)
        ghost_signals    = {}
        ghost_magnitudes = {}
        if self.use_ghost:
            for src_i, dst_i, gate_name in self.gate_pairs:
                key = gate_name.replace('→', '_')
                gated_out, alpha, mag = self.gates[key](embeddings[src_i], embeddings[dst_i])
                gated[src_i] = gated_out
                ghost_signals[gate_name]    = alpha
                ghost_magnitudes[gate_name] = mag
        else:
            for src_i, dst_i, gate_name in self.gate_pairs:
                ghost_signals[gate_name]    = 0.0
                ghost_magnitudes[gate_name] = 0.0

        combined = torch.cat(gated, dim=-1)
        logits   = self.classifier(combined)

        if not return_audit:
            return logits

        # ── Chunk logit decomposition ──────────────────────────────────────────
        W = self.classifier.weight   # (n_classes, embed_dim * n_chunks)
        D = self.embed_dim
        chunk_contributions = {}
        for i in range(self.n_chunks):
            W_block = W[:, i*D:(i+1)*D]
            contrib = W_block @ gated[i].T                  # (2, N)
            disease_push = contrib[1] - contrib[0]
            chunk_contributions[f'Chunk{i}'] = {
                'disease_logit': round(float(contrib[1].mean()), 4),
                'healthy_logit': round(float(contrib[0].mean()), 4),
                'disease_push':  round(float(disease_push.mean()), 4),
            }

        # ── Order decomposition audit ──────────────────────────────────────────
        order_decomp = {}
        if self.use_order_decomp:
            for i in range(self.n_chunks):
                cname = f'Chunk{i}'
                # With MoE, use first sub-chunk's order gate as representative
                beta_key = f'{cname}_order_beta' if self.n_sub_chunks == 1 \
                           else f'{cname}_sub0_order_beta'
                lin_key  = f'{cname}_linear_norm' if self.n_sub_chunks == 1 \
                           else f'{cname}_sub0_linear_norm'
                nlin_key = f'{cname}_nonlinear_norm' if self.n_sub_chunks == 1 \
                           else f'{cname}_sub0_nonlinear_norm'

                beta        = all_norms.get(beta_key)
                lin_norm    = all_norms.get(lin_key)
                nonlin_norm = all_norms.get(nlin_key)
                if beta is not None:
                    effective_nonlin = beta * nonlin_norm
                    total    = lin_norm + effective_nonlin + 1e-8
                    lin_frac = lin_norm / total
                    order_decomp[cname] = {
                        'beta':           round(beta, 4),
                        'linear_frac':    round(float(lin_frac), 4),
                        'linear_norm':    round(lin_norm, 4),
                        'nonlinear_norm': round(nonlin_norm, 4),
                        'dominant':       'linear' if lin_frac > 0.5 else 'nonlinear',
                    }

        # ── Sub-chunk routing audit ────────────────────────────────────────────
        sub_chunk_routing = {}
        if self.n_sub_chunks > 1:
            for i in range(self.n_chunks):
                cname = f'Chunk{i}'
                routing = {}
                for k in range(self.n_sub_chunks):
                    w = all_norms.get(f'{cname}_sub{k}_routing', 0.0)
                    routing[f'sub{k}'] = round(w, 4)
                dominant = all_norms.get(f'{cname}_dominant_sub', 0)
                sub_chunk_routing[cname] = {
                    'routing_weights': routing,
                    'dominant_sub':    f'sub{dominant}',
                }

        audit_dict = {
            'ghost_signals':       ghost_signals,
            'ghost_magnitudes':    ghost_magnitudes,
            'chunk_contributions': chunk_contributions,
            'chunk_norms':         all_norms,
            'order_decomp':        order_decomp,
            'sub_chunk_routing':   sub_chunk_routing,
        }
        return logits, audit_dict

    def get_all_gate_weights(self) -> dict:
        if not self.use_ghost:
            return {name: 0.0 for _, _, name in self.gate_pairs}
        return {
            name: self.gates[name.replace('→', '_')].get_gate_weight()
            for _, _, name in self.gate_pairs
        }

    def get_order_weights(self) -> dict:
        """Return converged β (order gate) per chunk. Only valid if use_order_decomp=True."""
        if not self.use_order_decomp:
            return {}
        if self.n_sub_chunks > 1:
            return {
                f'Chunk{i}': torch.sigmoid(chunk.order_gates[0]).item()
                for i, chunk in enumerate(self.chunks)
            }
        return {
            f'Chunk{i}': torch.sigmoid(chunk.order_gate).item()
            for i, chunk in enumerate(self.chunks)
        }

    def get_gate_l1_loss(self) -> torch.Tensor:
        """L1 regularisation on ghost gates, order gates, and router entropy."""
        losses = []

        if self.use_ghost:
            logits = [self.gates[name.replace('→', '_')]._last_logit
                      for _, _, name in self.gate_pairs
                      if self.gates[name.replace('→', '_')]._last_logit is not None]
            if logits:
                losses.append(F.softplus(torch.cat(logits, dim=-1)).mean())

        if self.use_order_decomp:
            if self.n_sub_chunks > 1:
                order_betas = torch.stack([
                    gate for chunk in self.chunks for gate in chunk.order_gates
                ])
            else:
                order_betas = torch.stack([chunk.order_gate for chunk in self.chunks])
            losses.append(F.softplus(order_betas).sum())

        return torch.stack(losses).sum() if losses else torch.tensor(0.0)
