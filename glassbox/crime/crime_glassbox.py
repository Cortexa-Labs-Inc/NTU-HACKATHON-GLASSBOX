"""
CrimeGlassboxNet — GlassboxNetV2 extended for multi-class crime detection.

Adds:
  get_class_pair_contributions(x, pred_class, true_class)
    → per-chunk blame scores for a specific misclassification pair

  get_embeddings(x)
    → raw and gated chunk embeddings (for failure-mode clustering)

The base GlassboxNetV2 is reused unchanged for chunked forward pass,
ghost gates, order decomposition, and standard audit dict.
"""

import torch
from model.glassbox_net_v2 import GlassboxNetV2


class CrimeGlassboxNet(GlassboxNetV2):
    """
    GlassboxNetV2 with multi-class chunk attribution.

    chunk_names: list of human-readable names for each chunk
                 (e.g. ['Motion', 'Appearance', 'Spatial', 'Temporal'])
    All other args pass through to GlassboxNetV2.
    """

    def __init__(self, chunk_sizes: list, chunk_names: list = None,
                 embed_dim: int = 16, n_classes: int = 14,
                 use_ghost: bool = True, use_order_decomp: bool = True,
                 n_sub_chunks: int = 1):
        super().__init__(
            chunk_sizes=chunk_sizes,
            embed_dim=embed_dim,
            n_classes=n_classes,
            use_ghost=use_ghost,
            use_order_decomp=use_order_decomp,
            n_sub_chunks=n_sub_chunks,
        )
        n = len(chunk_sizes)
        self.chunk_names = chunk_names or [f'Chunk{i}' for i in range(n)]

    # ── Override forward to remap generic 'Chunk{i}' keys → semantic names ────

    def forward(self, x, return_audit=False):
        """
        Identical to GlassboxNetV2.forward but renames audit dict keys:
          'Chunk0' → self.chunk_names[0]  (e.g. 'Motion')
          'Chunk1' → self.chunk_names[1]  (e.g. 'Appearance')
        in chunk_contributions and order_decomp.

        This preserves the exact V2 logit decomposition guarantee while giving
        the surveillance-domain names throughout the full audit pipeline.
        """
        if not return_audit:
            return super().forward(x, return_audit=False)

        logits, audit = super().forward(x, return_audit=True)

        # Remap Chunk{i} → semantic chunk name in both sub-dicts
        audit['chunk_contributions'] = {
            self.chunk_names[i]: v
            for i, (_, v) in enumerate(audit['chunk_contributions'].items())
        }
        if audit.get('order_decomp'):
            audit['order_decomp'] = {
                self.chunk_names[i]: v
                for i, (_, v) in enumerate(audit['order_decomp'].items())
            }
        if audit.get('sub_chunk_routing'):
            audit['sub_chunk_routing'] = {
                self.chunk_names[i]: v
                for i, (_, v) in enumerate(audit['sub_chunk_routing'].items())
            }

        return logits, audit

    # ── Internal: capture gated embeddings ────────────────────────────────────

    def _get_gated_embeddings(self, x: torch.Tensor) -> list:
        """
        Re-run the forward pass and return gated chunk embeddings.
        Returns [gated_0, gated_1, ...] each shape (N, embed_dim).
        """
        # Chunk forward passes
        embeddings = []
        offset = 0
        for chunk, sz in zip(self.chunks, self.chunk_sizes):
            xi = x[:, offset:offset + sz]
            emb, _ = chunk(xi)
            embeddings.append(emb)
            offset += sz

        # Ghost gates
        gated = list(embeddings)
        if self.use_ghost:
            for src_i, dst_i, gate_name in self.gate_pairs:
                key = gate_name.replace('→', '_')
                gated_out, _, _ = self.gates[key](embeddings[src_i], embeddings[dst_i])
                gated[src_i] = gated_out
        return gated

    # ── Public: multi-class chunk blame attribution ───────────────────────────

    def get_class_pair_contributions(
        self, x: torch.Tensor, pred_class: int, true_class: int
    ) -> dict:
        """
        For each chunk compute:
          pred_push  = how much this chunk pushed toward the (wrong) predicted class
          true_push  = how much this chunk pushed toward the (correct) true class
          blame      = pred_push − true_push  (positive → this chunk caused the error)

        Returns
        -------
        {chunk_name: {'pred_push': float, 'true_push': float, 'blame': float}}
        """
        self.eval()
        with torch.no_grad():
            gated = self._get_gated_embeddings(x)
            W = self.classifier.weight   # (n_classes, embed_dim * n_chunks)
            D = self.embed_dim

            result = {}
            for i, name in enumerate(self.chunk_names):
                W_block = W[:, i * D:(i + 1) * D]     # (n_classes, D)
                contrib = (W_block @ gated[i].T)        # (n_classes, N)
                pred_push = float(contrib[pred_class].mean())
                true_push = float(contrib[true_class].mean())
                result[name] = {
                    'pred_push': round(pred_push, 4),
                    'true_push': round(true_push, 4),
                    'blame':     round(pred_push - true_push, 4),
                }
        return result

    def get_embeddings(self, x: torch.Tensor) -> dict:
        """
        Return raw chunk embeddings (before gating) and gated embeddings.
        Useful for failure-mode clustering — we cluster in gated-embed space.
        """
        self.eval()
        with torch.no_grad():
            raw_embeddings = []
            offset = 0
            for chunk, sz in zip(self.chunks, self.chunk_sizes):
                xi = x[:, offset:offset + sz]
                emb, _ = chunk(xi)
                raw_embeddings.append(emb)
                offset += sz

            gated = self._get_gated_embeddings(x)
            combined_gated = torch.cat(gated, dim=-1)   # (N, embed_dim * n_chunks)

        return {
            'raw':    [e.cpu().numpy() for e in raw_embeddings],
            'gated':  [g.cpu().numpy() for g in gated],
            'combined': combined_gated.cpu().numpy(),
        }
