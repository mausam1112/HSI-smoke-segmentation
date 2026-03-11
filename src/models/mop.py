"""
Mixture of Prototypes (MoP) network for hyperspectral smoke segmentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BandSplitModule(nn.Module):
    """
    Splits the C input bands into G groups of size C//G and projects
    each group independently to `band_dim` features via a shared-weight
    (group-wise) 1×1 conv.  Keeping groups independent prevents
    cross-band contamination that occurs when all bands are fused by a
    single conv before spectral weighting.

    Parameters
    ----------
    in_channels : int  — total spectral bands C
    n_groups    : int  — number of band groups G
    band_dim    : int  — per-group feature dimension D
    """

    def __init__(self, in_channels: int, n_groups: int, band_dim: int) -> None:
        super().__init__()
        assert in_channels % n_groups == 0, (
            f"in_channels ({in_channels}) must be divisible by n_groups ({n_groups})"
        )
        self.n_groups = n_groups
        self.group_size = in_channels // n_groups
        self.band_dim = band_dim

        # One independent projection per group (no weight sharing across groups)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.group_size, band_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(band_dim),
                nn.GELU(),
            )
            for _ in range(n_groups)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W) → (B, G, D, H, W)
        """
        # B, C, H, W = x.shape
        groups = x.chunk(self.n_groups, dim=1)           # G × (B, C//G, H, W)
        out = torch.stack(
            [proj(g) for proj, g in zip(self.projections, groups)],
            dim=1,
        )                                                 # (B, G, D, H, W)
        return out


class PrototypeSpectralRepresentation(nn.Module):
    """
    Maintains K learnable prototype vectors of dimension D for each of
    the G band groups.  Each spatial position in the feature map is
    soft-assigned to the K prototypes via cosine similarity, producing a
    K-dimensional mixture coefficient for every pixel and group.

    The output replaces the raw D-dimensional features with a richer
    K-dimensional prototype mixture code that can represent multiple
    distinct spectral signatures (the "mixture" in MoP).

    Parameters
    ----------
    n_groups     : int  — G
    band_dim     : int  — D (input feature dim per group)
    n_prototypes : int  — K
    """

    def __init__(self, n_groups: int, band_dim: int, n_prototypes: int) -> None:
        super().__init__()
        self.n_groups = n_groups
        self.n_prototypes = n_prototypes

        # Learnable prototypes: (G, K, D)
        self.prototypes = nn.Parameter(
            torch.randn(n_groups, n_prototypes, band_dim)
        )
        nn.init.kaiming_normal_(
            self.prototypes.view(n_groups * n_prototypes, band_dim),
            mode="fan_out",
        )
        # Temperature for softmax over prototypes (learnable scalar)
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, G, D, H, W) → (B, G, K, H, W)   mixture assignment maps

        For each group g and spatial position (h,w) we compute:
            a_{g,k}(h,w) = softmax_k( cos_sim(x[g,:,h,w], proto[g,k]) / τ )
        """
        B, G, D, H, W = x.shape

        # Flatten spatial: (B, G, D, H*W) → (B*G, H*W, D)
        xf = x.view(B * G, D, H * W).permute(0, 2, 1)           # (BG, HW, D)

        # Normalise features and prototypes for cosine similarity
        xf_norm = F.normalize(xf, dim=-1)                        # (BG, HW, D)
        protos = self.prototypes.view(G, self.n_prototypes, D)   # (G, K, D)
        protos = protos.repeat(B, 1, 1)                          # (BG, K, D)
        protos_norm = F.normalize(protos, dim=-1)                 # (BG, K, D)

        # Cosine similarity: (BG, HW, K)
        sim = torch.bmm(xf_norm, protos_norm.permute(0, 2, 1))   # (BG, HW, K)

        # Soft assignment via temperature-scaled softmax
        tau = self.temperature.clamp(min=1e-4)
        attn = F.softmax(sim / tau, dim=-1)                       # (BG, HW, K)

        # Prototype mixture: (BG, HW, K) → (B, G, K, H, W)
        out = attn.permute(0, 2, 1).view(B, G, self.n_prototypes, H, W)
        return out


class DualLevelRouter(nn.Module):
    """
    Level-1 (global gate): from each group's mean prototype assignment,
        learn a scalar gate γ_g ∈ (0,1) that suppresses uninformative groups.

    Level-2 (local map): a 3×3 depth-wise conv over the K-channel map
        of each group produces a spatially varying attention map α_g(h,w).

    Final weight W_g(h,w) = γ_g · α_g(h,w)   (broadcast × spatial).

    The weighted prototype maps from all groups are concatenated to form
    the fused representation fed to the decoder.

    Parameters
    ----------
    n_groups     : int  — G
    n_prototypes : int  — K
    """

    def __init__(self, n_groups: int, n_prototypes: int) -> None:
        super().__init__()
        self.n_groups = n_groups
        self.n_prototypes = n_prototypes

        # Level-1: global gate per group (G scalars)
        self.global_gate = nn.Sequential(
            nn.Linear(n_prototypes, n_prototypes // 2 if n_prototypes > 2 else n_prototypes),
            nn.ReLU(inplace=True),
            nn.Linear(n_prototypes // 2 if n_prototypes > 2 else n_prototypes, 1),
            nn.Sigmoid(),
        )

        # Level-2: spatial attention per group
        # Depth-wise conv over the K channels of each group
        self.local_attn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(n_prototypes, n_prototypes, kernel_size=3, padding=1,
                          groups=n_prototypes, bias=False),
                nn.Conv2d(n_prototypes, 1, kernel_size=1, bias=False),
                nn.Sigmoid(),
            )
            for _ in range(n_groups)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, G, K, H, W)  — prototype assignment maps
        → (B, G*K, H, W)     — routed, weighted features
        """
        B, G, K, H, W = x.shape
        weighted_groups = []

        for g in range(G):
            feat_g = x[:, g, :, :, :]                   # (B, K, H, W)

            # Level-1: global gate γ_g
            global_ctx = feat_g.mean(dim=(2, 3))         # (B, K)
            gamma = self.global_gate(global_ctx)         # (B, 1)
            gamma = gamma.view(B, 1, 1, 1)               # broadcast to spatial

            # Level-2: local spatial attention α_g(h,w)
            alpha = self.local_attn[g](feat_g)           # (B, 1, H, W)

            # Combined weight
            weight = gamma * alpha                       # (B, 1, H, W)
            weighted_groups.append(feat_g * weight)     # (B, K, H, W)

        return torch.cat(weighted_groups, dim=1)         # (B, G*K, H, W)


# Decoder

def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding="same", bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class MoPDecoder(nn.Module):
    """
    Simple two-stage conv decoder that maps the routed features
    (B, G*K, H, W) to a binary segmentation map (B, num_classes, H, W).

    The spatial resolution is preserved throughout (no downsampling in MoP
    encoder), so the decoder is a pure refinement head.
    """

    def __init__(self, in_channels: int, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _conv_bn_relu(in_channels, embed_dim),
            _conv_bn_relu(embed_dim, embed_dim // 2),
            nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Full MoP network
class MixtureOfPrototypes(nn.Module):
    """
    Mixture of Prototypes (MoP) network.

    Implements the full pipeline from the paper:
      Band Split → Prototype Assignment → Dual-Level Router → Decode

    Parameters
    ----------
    in_channels  : int   — spectral bands C (default 25)
    num_classes  : int   — output channels (1 for binary)
    n_groups     : int   — G, band groups (default 5)
    band_dim     : int   — D, per-group feature dim (default 32)
    n_prototypes : int   — K, prototypes per group (default 8)
    embed_dim    : int   — decoder intermediate channels (default 64)
    """

    def __init__(
        self,
        in_channels: int = 25,
        num_classes: int = 1,
        n_groups: int = 5,
        band_dim: int = 32,
        n_prototypes: int = 8,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()

        self.n_groups = n_groups
        self.n_prototypes = n_prototypes

        self.bsm = BandSplitModule(in_channels, n_groups, band_dim)
        self.psr = PrototypeSpectralRepresentation(n_groups, band_dim, n_prototypes)
        self.dlr = DualLevelRouter(n_groups, n_prototypes)
        self.decoder = MoPDecoder(
            in_channels=n_groups * n_prototypes,
            embed_dim=embed_dim,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W)
        → (B, num_classes, H, W)  values in [0,1]
        """
        band_feats  = self.bsm(x)          # (B, G, D, H, W)
        proto_maps  = self.psr(band_feats)  # (B, G, K, H, W)
        routed      = self.dlr(proto_maps)  # (B, G*K, H, W)
        return self.decoder(routed)         # (B, num_classes, H, W)

    def model_summary(self, input_size: tuple) -> None:
        try:
            import torchinfo
            print("\n" + "=" * 70)
            print(f"Model Summary: {self.__class__.__name__}")
            print(f"  n_groups={self.n_groups}  n_prototypes={self.n_prototypes}")
            print("=" * 70)
            torchinfo.summary(self, input_size=input_size, depth=4,
                              col_names=["input_size", "output_size", "num_params", "trainable"])
            print("=" * 70 + "\n")
        except ImportError:
            print(self)
