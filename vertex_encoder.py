import torch
import torch.nn as nn

from lato.modules.pointclouds.pointnet import LocalPoolPointnet


class VoxelFeatureEncoder_active_pointnet(nn.Module):
    """PointNet-based encoder for active voxel features."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        scatter_type: str,
        n_blocks: int,
        resolution: int = 64,
        add_label: bool = False,
    ):
        super().__init__()
        self.pointnet = LocalPoolPointnet(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            scatter_type=scatter_type,
        )
        self.add_label = add_label
        self.resolution = resolution

        if add_label:
            self.label_emb = nn.Embedding(3, 16)
            self.fusion_mlp = nn.Sequential(
                nn.Linear(out_channels + 16, out_channels),
                nn.GELU(),
                nn.Linear(out_channels, out_channels),
            )

    def forward(
        self,
        p: torch.Tensor,
        sparse_coords: torch.Tensor,
        res: int | None = None,
        bbox_size=(-0.5, 0.5),
        voxel_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        res = self.resolution if res is None else res
        geo_feats = self.pointnet(p, sparse_coords, res=res, bbox_size=bbox_size)
        if voxel_label is None:
            return geo_feats
        label_feats = self.label_emb(voxel_label)
        return self.fusion_mlp(torch.cat([geo_feats, label_feats], dim=-1))


class ConnectionHead(nn.Module):
    """Small MLP head for edge or connection logits."""

    def __init__(self, channels: int, out_channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_channels = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
