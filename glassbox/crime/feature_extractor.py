"""
Multi-scale CNN feature extractor + end-to-end CrimeVisionGlassbox.

Architecture
------------
PNG image (3, H, W)
    │
    ▼  ResNet18 backbone (ImageNet pretrained, frozen by default)
    │
    ├─ Stage 1 → GAP → Linear(64,  proj_dim)  → Texture   chunk  [edges, textures]
    ├─ Stage 2 → GAP → Linear(128, proj_dim)  → Structure chunk  [parts, boundaries]
    ├─ Stage 3 → GAP → Linear(256, proj_dim)  → Context   chunk  [objects, scene]
    └─ Stage 4 → GAP → Linear(512, proj_dim)  → Semantic  chunk  [crime patterns]
    │
    ▼  concat → (N, 4 × proj_dim)
    │
    ▼  CrimeGlassboxNet (GlassboxNetV2)
       4 ChunkNets + C(4,2)=6 Ghost gates + order decomp β per chunk
       Exact logit decomposition → per-chunk blame attribution
    │
    ▼  14-class softmax

The self-healing loop operates entirely in the 4×proj_dim feature space
(after CNN extraction).  Gaussian perturbation around failure centroids
generates synthetic feature vectors — no image synthesis needed.

Key interpretability mappings:
  Ghost gate C0→C1  (Texture→Structure):
    "Texture patterns abnormally coupled with structural representations"
    e.g. unusual ground texture influencing body-part detection
  Ghost gate C2→C3  (Context→Semantic):
    "Object-level context distorted high-level semantic interpretation"
    e.g. crowd context masking individual criminal action pattern
"""

import torch
import torch.nn as nn

from crime.crime_glassbox import CrimeGlassboxNet

# Chunk names mapped to surveillance-domain semantics
CHUNK_NAMES = ['Texture', 'Structure', 'Context', 'Semantic']
STAGE_DIMS  = [64, 128, 256, 512]   # ResNet18 stage channel counts


def _conv_block(in_ch, out_ch, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class TinyCNNExtractor(nn.Module):
    """
    Lightweight 4-stage CNN — no pretrained download needed.

    64×64 input → 4 progressively deeper feature maps → GAP → proj_dim each.
    ~200K params total, trains fast on CPU.

    Stage dims: 32 → 64 → 128 → 256  (matches CHUNK_NAMES semantics)
    """

    STAGE_DIMS = [32, 64, 128, 256]

    def __init__(self, proj_dim: int = 32):
        super().__init__()
        self.proj_dim = proj_dim
        self.stem   = _conv_block(3,   32, stride=2)   # 64→32
        self.stage1 = _conv_block(32,  64, stride=2)   # 32→16  Texture
        self.stage2 = _conv_block(64, 128, stride=2)   # 16→8   Structure
        self.stage3 = _conv_block(128,256, stride=2)   # 8→4    Context
        self.stage4 = _conv_block(256,256, stride=2)   # 4→2    Semantic
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, proj_dim, bias=True),
                nn.LayerNorm(proj_dim),
                nn.ReLU(inplace=True),
            )
            for d in [64, 128, 256, 256]
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        feats = []
        for stage, proj in zip(
            [self.stage1, self.stage2, self.stage3, self.stage4], self.proj
        ):
            h = stage(h)
            feats.append(proj(self.gap(h).flatten(1)))
        return torch.cat(feats, dim=1)   # (N, 4 * proj_dim)


class MultiScaleCNNExtractor(nn.Module):
    """
    ResNet18 backbone split into 4 residual stages.

    Each stage output is global-average-pooled and projected to proj_dim,
    producing one Glassbox chunk.

    Parameters
    ----------
    proj_dim        : output dimension per chunk (default 32)
    pretrained      : load ImageNet weights (default True)
    freeze_backbone : if True, backbone weights are frozen — only the linear
                      projections are trained during fine-tuning (default True)
    """

    def __init__(self, proj_dim: int = 32, pretrained: bool = True,
                 freeze_backbone: bool = True):
        super().__init__()
        self.proj_dim = proj_dim

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # Stage 0: conv1 + bn1 + relu + maxpool
        self.stage0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        # Residual stages
        self.stage1 = backbone.layer1   # out: (N, 64,  H/4,  W/4)
        self.stage2 = backbone.layer2   # out: (N, 128, H/8,  W/8)
        self.stage3 = backbone.layer3   # out: (N, 256, H/16, W/16)
        self.stage4 = backbone.layer4   # out: (N, 512, H/32, W/32)

        if freeze_backbone:
            for p in [self.stage0, self.stage1, self.stage2,
                      self.stage3, self.stage4]:
                for param in p.parameters():
                    param.requires_grad = False

        self.gap = nn.AdaptiveAvgPool2d(1)

        # One linear projection per stage → proj_dim
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, proj_dim, bias=True),
                nn.LayerNorm(proj_dim),
                nn.ReLU(inplace=True),
            )
            for d in STAGE_DIMS
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, 3, H, W) normalised image tensor

        Returns
        -------
        features : (N, 4 * proj_dim) — concatenated multi-scale feature chunks
        """
        h = self.stage0(x)

        feats = []
        for stage, proj in zip(
            [self.stage1, self.stage2, self.stage3, self.stage4],
            self.proj,
        ):
            h = stage(h)
            pooled = self.gap(h).flatten(1)   # (N, stage_dim)
            feats.append(proj(pooled))          # (N, proj_dim)

        return torch.cat(feats, dim=1)          # (N, 4 * proj_dim)

    def forward_chunks(self, x: torch.Tensor) -> list:
        """
        Same as forward but returns a list of 4 tensors (one per chunk)
        rather than a concatenated tensor.  Used for direct inspection.
        """
        h = self.stage0(x)
        chunks = []
        for stage, proj in zip(
            [self.stage1, self.stage2, self.stage3, self.stage4],
            self.proj,
        ):
            h = stage(h)
            chunks.append(proj(self.gap(h).flatten(1)))
        return chunks


class CrimeVisionGlassbox(nn.Module):
    """
    End-to-end model: PNG image → interpretable 14-class crime prediction.

    Parameters
    ----------
    n_classes       : number of crime classes (default 14)
    proj_dim        : CNN projection dimension per chunk (default 32)
    embed_dim       : Glassbox embedding dimension per chunk (default 16)
    backbone        : 'tiny' (default, fast CPU, no download) or 'resnet18'
    pretrained      : use ImageNet-pretrained ResNet18 (only when backbone='resnet18')
    freeze_backbone : freeze backbone weights (only when backbone='resnet18')
    use_ghost       : enable Ghost Signal Gates
    use_order_decomp: enable order decomposition β gates per chunk
    n_sub_chunks    : number of sub-experts per named chunk (1 = off, 3 = recommended)
    """

    def __init__(
        self,
        n_classes:       int   = 14,
        proj_dim:        int   = 32,
        embed_dim:       int   = 16,
        backbone:        str   = 'tiny',
        pretrained:      bool  = True,
        freeze_backbone: bool  = True,
        use_ghost:       bool  = True,
        use_order_decomp: bool = True,
        n_sub_chunks:    int   = 1,
    ):
        super().__init__()
        self.proj_dim  = proj_dim
        self.n_classes = n_classes

        if backbone == 'resnet18':
            self.extractor = MultiScaleCNNExtractor(
                proj_dim=proj_dim,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
            )
        else:
            self.extractor = TinyCNNExtractor(proj_dim=proj_dim)
        self.glassbox = CrimeGlassboxNet(
            chunk_sizes=[proj_dim] * 4,
            chunk_names=CHUNK_NAMES,
            embed_dim=embed_dim,
            n_classes=n_classes,
            use_ghost=use_ghost,
            use_order_decomp=use_order_decomp,
            n_sub_chunks=n_sub_chunks,
        )

    def forward(self, x: torch.Tensor, return_audit: bool = False):
        """
        Parameters
        ----------
        x            : (N, 3, H, W) image tensor
        return_audit : if True, also return Glassbox audit dict

        Returns
        -------
        logits     : (N, n_classes)
        audit_dict : (only if return_audit=True)  — full Glassbox audit including:
            chunk_contributions   {Texture/Structure/Context/Semantic: {pred_push, true_push, ...}}
            ghost_signals         {C0→C1, ..., C2→C3: alpha_float}
            ghost_magnitudes      {gate: magnitude}
            order_decomp          {chunk: {beta, linear_frac, dominant}}
            chunk_norms           {chunk_layer: norm}
        """
        features = self.extractor(x)          # (N, 4 * proj_dim)
        return self.glassbox(features, return_audit=return_audit)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the concatenated feature vector (N, 4 * proj_dim) for a batch.
        Used by the self-healing failure detector to collect feature vectors
        of misclassified samples for K-means clustering.
        """
        with torch.no_grad():
            return self.extractor(x)

    def get_class_pair_contributions(
        self, x: torch.Tensor, pred_class: int, true_class: int
    ) -> dict:
        """
        Multi-class chunk blame attribution.
        Delegates to CrimeGlassboxNet after feature extraction.
        """
        features = self.extractor(x)
        return self.glassbox.get_class_pair_contributions(features, pred_class, true_class)

    def get_embeddings(self, x: torch.Tensor) -> dict:
        """Return raw + gated chunk embeddings for failure analysis."""
        features = self.extractor(x)
        return self.glassbox.get_embeddings(features)

    def get_gate_l1_loss(self) -> torch.Tensor:
        return self.glassbox.get_gate_l1_loss()

    @property
    def chunk_names(self):
        return self.glassbox.chunk_names

    @property
    def gate_pairs(self):
        return self.glassbox.gate_pairs
