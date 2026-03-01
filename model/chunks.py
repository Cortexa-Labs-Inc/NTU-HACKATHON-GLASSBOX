import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mlp(input_dim: int, hidden_dims: list, output_dim: int) -> nn.Sequential:
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class ChunkNet(nn.Module):
    """
    Deep MLP subnetwork for one semantically-grouped feature chunk.

    Modes
    -----
    n_sub_chunks == 1 (default):
        Single MLP pathway — original behaviour.

    n_sub_chunks > 1  — Mixture of Sub-Experts (MoSE):
        K parallel sub-MLPs process the same chunk input.
        A learned router (softmax over K) produces per-sample mixing weights.
        Output = weighted sum of K sub-embeddings.

        The router discovers mathematical sub-patterns within the named chunk
        that the 4 top-level names don't capture — e.g. within 'Texture':
          sub0 might specialise in high-frequency motion
          sub1 might specialise in edge-density uniformity
          sub2 might specialise in periodic spatial patterns
        You don't define these; the model finds them.
        The routing weights (exposed in activation_norms) tell you which
        sub-pattern fired for each frame.

    use_order_decomp:
        Splits each sub-expert (or the single MLP) into a linear 1st-order
        path and a nonlinear nth-order path gated by learned β.
    """

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 chunk_name: str, use_order_decomp: bool = False,
                 n_sub_chunks: int = 1):
        super().__init__()
        self.chunk_name      = chunk_name
        self.use_order_decomp = use_order_decomp
        self.n_sub_chunks    = n_sub_chunks

        if n_sub_chunks > 1:
            # ── Mixture of Sub-Experts ────────────────────────────────────────
            self.sub_nets = nn.ModuleList([
                _make_mlp(input_dim, hidden_dims, output_dim)
                for _ in range(n_sub_chunks)
            ])
            # Router: input → K logits → softmax weights
            self.router = nn.Linear(input_dim, n_sub_chunks)

            if use_order_decomp:
                self.linear_paths = nn.ModuleList([
                    nn.Linear(input_dim, output_dim, bias=True)
                    for _ in range(n_sub_chunks)
                ])
                self.order_gates = nn.ParameterList([
                    nn.Parameter(torch.tensor(0.0))
                    for _ in range(n_sub_chunks)
                ])
        else:
            # ── Single pathway (original) ─────────────────────────────────────
            self.network = _make_mlp(input_dim, hidden_dims, output_dim)

            if use_order_decomp:
                self.linear_path = nn.Linear(input_dim, output_dim, bias=True)
                self.order_gate  = nn.Parameter(torch.tensor(0.0))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_single(self, x, norms: dict, suffix: str = ''):
        """Run one MLP (or order-decomp pair) and record norms."""
        net = self.network if not suffix else self.sub_nets[int(suffix.lstrip('_sub'))]

        if self.use_order_decomp:
            lin_layer  = self.linear_path if not suffix else self.linear_paths[int(suffix.lstrip('_sub'))]
            gate_param = self.order_gate  if not suffix else self.order_gates[int(suffix.lstrip('_sub'))]

            emb_lin    = lin_layer(x)
            emb_nonlin = self._run_mlp(net, x, norms, suffix)
            beta       = torch.sigmoid(gate_param)
            emb        = emb_lin + beta * emb_nonlin

            key = f'{self.chunk_name}{suffix}'
            norms[f'{key}_order_beta']       = beta.item()
            norms[f'{key}_linear_norm']      = emb_lin.norm(dim=-1).mean().item()
            norms[f'{key}_nonlinear_norm']   = emb_nonlin.norm(dim=-1).mean().item()
        else:
            emb = self._run_mlp(net, x, norms, suffix)

        return emb

    @staticmethod
    def _run_mlp(net, x, norms: dict, suffix: str):
        h = x
        layer_num = 0
        for module in net:
            h = module(h)
            if isinstance(module, nn.ReLU):
                norms[f'layer{layer_num}{suffix}'] = h.norm(dim=-1).mean().item()
                layer_num += 1
        return h

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x):
        """
        Returns
        -------
        embedding        : (N, output_dim)
        activation_norms : dict — layer norms, order-gate info, sub-chunk routing weights
        """
        norms = {}

        if self.n_sub_chunks > 1:
            # Router weights: (N, K)
            routing_logits  = self.router(x)
            routing_weights = F.softmax(routing_logits, dim=-1)   # (N, K)

            # Each sub-expert
            sub_embs = []
            for k in range(self.n_sub_chunks):
                suffix = f'_sub{k}'
                if self.use_order_decomp:
                    emb_lin    = self.linear_paths[k](x)
                    emb_nonlin = self._run_mlp(self.sub_nets[k], x, norms, suffix)
                    beta       = torch.sigmoid(self.order_gates[k])
                    emb        = emb_lin + beta * emb_nonlin
                    key = f'{self.chunk_name}{suffix}'
                    norms[f'{key}_order_beta']     = beta.item()
                    norms[f'{key}_linear_norm']    = emb_lin.norm(dim=-1).mean().item()
                    norms[f'{key}_nonlinear_norm'] = emb_nonlin.norm(dim=-1).mean().item()
                else:
                    emb = self._run_mlp(self.sub_nets[k], x, norms, suffix)
                sub_embs.append(emb)

            sub_embs_t  = torch.stack(sub_embs, dim=1)          # (N, K, D)
            embedding   = (routing_weights.unsqueeze(-1) * sub_embs_t).sum(dim=1)  # (N, D)

            # Expose mean routing weight per sub-chunk (scalar per sub-chunk)
            mean_weights = routing_weights.mean(dim=0)           # (K,)
            for k in range(self.n_sub_chunks):
                norms[f'{self.chunk_name}_sub{k}_routing'] = round(mean_weights[k].item(), 4)

            # Also store per-sample dominant sub-chunk
            dominant_k = routing_weights.argmax(dim=1)           # (N,)
            norms[f'{self.chunk_name}_dominant_sub'] = int(dominant_k.mode().values.item())

        else:
            embedding = self._run_single(x, norms)

        return embedding, norms
