from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules import sparse as sp
from ...modules.sparse import SparseTensor
from ...modules.sparse.linear import SparseLinear
from ...modules.sparse.nonlinearity import SparseGELU
from ...modules.utils import DiagonalGaussianDistribution, convert_module_to_f16, convert_module_to_f32, zero_module
from .base import SparseTransformerBase, SparseTransformerCrossBase
from .utils import SparseResBlock3d


class SparsePredictionHead(nn.Module):
    def __init__(self, channels: int, out_channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_channels = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            SparseLinear(channels, hidden_channels),
            SparseGELU(approximate="tanh"),
            SparseLinear(hidden_channels, out_channels),
        )

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        return self.mlp(x)


class SparseVertexSubdivideBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        resolution: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32,
        heads_num: int = 4,
        using_attn: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.out_resolution = resolution * 2
        self.out_channels = out_channels

        self.act_layers = nn.Sequential(
            sp.SparseGroupNorm32(num_groups, channels),
            sp.SparseSiLU(),
        )
        self.sub = sp.SparseSubdivide_attn(channels, heads_num) if using_attn else sp.SparseSubdivide()
        self.out_layers = nn.Sequential(
            sp.SparseConv3d(channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}"),
            sp.SparseGroupNorm32(num_groups, self.out_channels),
            sp.SparseSiLU(),
            zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}")),
        )
        self.skip_connection = (
            nn.Identity()
            if self.out_channels == channels
            else sp.SparseConv3d(channels, self.out_channels, 1, indice_key=f"res_{self.out_resolution}")
        )
        self.pruning_head = SparsePredictionHead(self.out_channels, out_channels=1)

    def forward(
        self,
        x: sp.SparseTensor,
        pruning: bool = False,
        training: bool = True,
        threshold: float = 0.5,
        force_no_prune: bool = False,
    ) -> Tuple[sp.SparseTensor, Optional[sp.SparseTensor]]:
        h = self.act_layers(x)
        h = self.sub(h)
        x = self.sub(x)
        h = self.out_layers(h)
        h = h + self.skip_connection(x)

        if not pruning:
            return h, None

        occ_prob = self.pruning_head(h)
        if not training and not force_no_prune:
            scores = torch.sigmoid(occ_prob.feats).squeeze(-1)
            if scores.numel() % 8 != 0:
                occ_mask = scores >= threshold
            else:
                grouped_scores = scores.view(-1, 8)
                grouped_mask = grouped_scores >= threshold
                empty_parent_mask = grouped_mask.sum(dim=1) == 0
                if empty_parent_mask.any():
                    _, top_indices = torch.topk(grouped_scores[empty_parent_mask], k=1, dim=1)
                    parent_rows = torch.nonzero(empty_parent_mask, as_tuple=True)[0].unsqueeze(1)
                    grouped_mask[parent_rows, top_indices] = True
                occ_mask = grouped_mask.view(-1)
            h = sp.SparseTensor(feats=h.feats[occ_mask], coords=h.coords[occ_mask])

        return h, occ_prob


class SparseVertexDecoderBlock(nn.Module):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        using_subdivide: bool = False,
        using_attn: bool = False,
    ):
        super().__init__()
        del num_blocks, num_heads, num_head_channels, mlp_ratio, attn_mode, window_size, pe_mode, use_checkpoint, qk_rms_norm
        self.using_subdivide = using_subdivide

        if using_subdivide:
            self.upsample = SparseVertexSubdivideBlock3d(
                channels=in_channels,
                resolution=resolution,
                out_channels=out_channels,
                num_groups=32,
                using_attn=using_attn,
            )
        else:
            self.upsample = SparseResBlock3d(
                channels=in_channels,
                out_channels=model_channels // 2,
                downsample=False,
                upsample=True,
            )

        if use_fp16:
            self.convert_to_fp16()

    def forward(
        self,
        x: sp.SparseTensor,
        context: Optional[sp.SparseTensor] = None,
        pruning: bool = False,
        training: bool = False,
        threshold: float = 0.5,
        force_no_prune: bool = False,
    ) -> Tuple[sp.SparseTensor, Optional[sp.SparseTensor]]:
        del context
        h = x.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        if self.using_subdivide:
            return self.upsample(
                h,
                pruning=pruning,
                training=training,
                threshold=threshold,
                force_no_prune=force_no_prune,
            )
        return self.upsample(h), None

    def convert_to_fp16(self):
        convert_module_to_f16(self.upsample)

    def convert_to_fp32(self):
        convert_module_to_f32(self.upsample)


class VoxelVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        encoder_blocks: Optional[List[Dict]] = None,
        decoder_blocks_vtx: Optional[List[Dict]] = None,
        num_heads: int = 8,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        attn_mode: str = "swin",
        window_size: int = 8,
        pe_mode: str = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = True,
        qk_rms_norm: bool = False,
        latent_dim: int = 8,
        using_subdivide: bool = True,
        using_attn: bool = False,
        ctx_channels: int = 768,
        attn_first: bool = True,
        pred_direction: bool = False,
    ):
        super().__init__()
        del in_channels, num_heads, ctx_channels, attn_first
        encoder_blocks = encoder_blocks or []
        decoder_blocks_vtx = decoder_blocks_vtx or []
        if not encoder_blocks:
            raise ValueError("encoder_blocks must contain at least one block config.")
        if not decoder_blocks_vtx:
            raise ValueError("decoder_blocks_vtx must contain at least one block config.")

        self.latent_dim = latent_dim
        self.using_subdivide = using_subdivide
        self.using_attn = using_attn
        self.decoder_blocks_vtx = decoder_blocks_vtx
        self.pred_direction = pred_direction

        enc_config = encoder_blocks[0]
        self.encoder = SparseTransformerBase(
            in_channels=enc_config["in_channels"],
            model_channels=enc_config["model_channels"],
            num_blocks=enc_config["num_blocks"],
            num_heads=enc_config["num_heads"],
            num_head_channels=num_head_channels,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            mlp_ratio=mlp_ratio,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        current_channels = enc_config["out_channels"]

        self.latent_expander = SparseTransformerBase(
            in_channels=latent_dim,
            model_channels=512,
            num_blocks=8,
            num_heads=8,
            num_head_channels=num_head_channels,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            mlp_ratio=mlp_ratio,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        self.vtx_proj = sp.SparseLinear(512, decoder_blocks_vtx[0]["in_channels"])
        self.out_layer = sp.SparseLinear(current_channels, 2 * latent_dim)
        self.vtx_head_64 = SparsePredictionHead(512, out_channels=1)

        self.decoder_vtx = nn.ModuleList(
            [
                SparseVertexDecoderBlock(
                    resolution=config["resolution"],
                    model_channels=config["model_channels"],
                    in_channels=config["in_channels"],
                    out_channels=config["out_channels"],
                    num_blocks=config["num_blocks"],
                    num_heads=config["num_heads"],
                    num_head_channels=num_head_channels,
                    mlp_ratio=mlp_ratio,
                    attn_mode=attn_mode,
                    window_size=window_size,
                    pe_mode=pe_mode,
                    use_fp16=use_fp16,
                    use_checkpoint=use_checkpoint,
                    qk_rms_norm=qk_rms_norm,
                    using_subdivide=using_subdivide,
                    using_attn=using_attn,
                )
                for config in decoder_blocks_vtx
            ]
        )
        self.latent_proj = nn.ModuleList([sp.SparseLinear(latent_dim, config["context_channels"]) for config in decoder_blocks_vtx])
        self.decoder_vtx_ca = nn.ModuleList(
            [
                SparseTransformerCrossBase(
                    in_channels=config["out_channels"],
                    model_channels=config["model_channels"],
                    context_channels=config["context_channels"],
                    num_blocks=config["num_blocks"],
                    num_heads=config["num_heads"],
                    num_head_channels=num_head_channels,
                    mlp_ratio=mlp_ratio,
                    attn_mode=attn_mode,
                    window_size=window_size,
                    pe_mode=pe_mode,
                    use_fp16=use_fp16,
                    use_checkpoint=use_checkpoint,
                    qk_rms_norm=qk_rms_norm,
                )
                for config in decoder_blocks_vtx
            ]
        )

        if use_fp16:
            self.convert_to_fp16()

    @staticmethod
    def _flatten_coords(coords_4d: torch.Tensor) -> torch.Tensor:
        coords_4d = coords_4d.long()
        return (
            coords_4d[:, 0] * (1024 ** 3)
            + coords_4d[:, 1] * (1024 ** 2)
            + coords_4d[:, 2] * 1024
            + coords_4d[:, 3]
        )

    def encode(
        self,
        x: sp.SparseTensor,
        ctx: Optional[SparseTensor] = None,
        sample_posterior: bool = True,
        prune_active: bool = False,
    ) -> Tuple[sp.SparseTensor, DiagonalGaussianDistribution]:
        del ctx
        h = self.encoder(x)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        if prune_active:
            return h

        posterior = DiagonalGaussianDistribution(h.feats, feat_dim=-1)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return h.replace(z), posterior

    def decode(
        self,
        latent_: sp.SparseTensor,
        gt_vertex_voxels_list: Optional[List[sp.SparseTensor]] = None,
        training: bool = True,
        pruning: bool = True,
        sample_ratio: float = 0.0,
        inference_threshold: float = 0.2,
        vis_last_layer: bool = True,
    ) -> List[Dict]:
        del pruning, sample_ratio
        latent = self.latent_expander(latent_)
        results = []

        if not training:
            vtx_probs = self.vtx_head_64(latent)
            scores = vtx_probs.feats[:, 0]
            vertex_mask = scores > inference_threshold
            if vertex_mask.sum() == 0 and scores.numel() > 0:
                _, top_indices = torch.topk(scores, k=min(2, scores.numel()))
                vertex_mask[top_indices] = True

            vertex_x = sp.SparseTensor(
                feats=latent.feats[vertex_mask],
                coords=latent.coords[vertex_mask],
            )
            vertex_x = self.vtx_proj(vertex_x)
            results.append(
                {
                    "coords": vtx_probs.coords[..., 1:],
                    "feats": vtx_probs.feats,
                    "sp_tensor": vtx_probs,
                    "vertex_mask": vertex_mask,
                    "vtx_sp": vertex_x,
                }
            )
        else:
            if gt_vertex_voxels_list is None:
                raise ValueError("gt_vertex_voxels_list is required during training.")
            vtx_probs = self.vtx_head_64(latent)
            gt_vertex_coords = (
                gt_vertex_voxels_list[0].coords
                if hasattr(gt_vertex_voxels_list[0], "coords")
                else gt_vertex_voxels_list[0]
            )
            vertex_mask = torch.isin(self._flatten_coords(latent.coords), self._flatten_coords(gt_vertex_coords))
            vertex_x = sp.SparseTensor(feats=latent.feats[vertex_mask], coords=latent.coords[vertex_mask])
            vertex_x = self.vtx_proj(vertex_x)
            results.append(
                {
                    "vtx_coords_3d": vtx_probs.coords,
                    "vtx_feats": vtx_probs.feats,
                    "vertex_mask": vertex_mask,
                    "vertex_gt_coords": gt_vertex_coords,
                }
            )

        for i, _ in enumerate(self.decoder_vtx):
            is_last_layer = i == len(self.decoder_vtx) - 1
            force_no_prune = (not training) and is_last_layer and vis_last_layer

            vertex_x, vertex_occ_probs = self.decoder_vtx[i](
                vertex_x,
                context=None,
                pruning=True,
                training=training,
                threshold=inference_threshold,
                force_no_prune=force_no_prune,
            )
            vertex_x = self.decoder_vtx_ca[i](x=vertex_x, context=self.latent_proj[i](latent_))

            if not training:
                vertex_result = {
                    "coords": vertex_x.coords[..., 1:],
                    "coords_4d": vertex_x.coords,
                    "feats": vertex_x.feats,
                    "sp_tensor": vertex_x,
                }
                if is_last_layer and vertex_occ_probs is not None:
                    vertex_result["occ_probs"] = vertex_occ_probs.feats
                results.append({"vertex": vertex_result})
                continue

            gt_vertex_coords = (
                gt_vertex_voxels_list[i + 1].coords
                if hasattr(gt_vertex_voxels_list[i + 1], "coords")
                else gt_vertex_voxels_list[i + 1]
            )
            vertex_isin_mask = torch.isin(self._flatten_coords(vertex_x.coords), self._flatten_coords(gt_vertex_coords))
            vertex_prune_labels = vertex_isin_mask.float()
            vertex_x = sp.SparseTensor(
                feats=vertex_x.feats[vertex_prune_labels.bool()],
                coords=vertex_x.coords[vertex_prune_labels.bool()],
            )
            results.append(
                {
                    "vertex": {
                        "coords": vertex_x.coords,
                        "feats": vertex_x.feats,
                        "occ_probs": vertex_occ_probs.feats,
                        "occ_coords": vertex_occ_probs.coords,
                        "prune_labels": vertex_prune_labels,
                        "sp_tensor": vertex_x,
                        "coords_4d": vertex_x.coords,
                        "gt_coords": gt_vertex_coords,
                        "pred_mask": vertex_prune_labels.bool(),
                    }
                }
            )

        return results

    def forward(
        self,
        sparse_input: sp.SparseTensor,
        gt_vertex_voxels_list: Optional[List[sp.SparseTensor]] = None,
        training: bool = True,
        sample_ratio: float = 0.0,
    ):
        latent_64, posterior = self.encode(sparse_input)
        results = self.decode(
            latent_64,
            gt_vertex_voxels_list=gt_vertex_voxels_list,
            training=training,
            sample_ratio=sample_ratio,
        )
        return results, posterior, latent_64

    def convert_to_fp16(self):
        self.encoder.apply(lambda m: m.convert_to_fp16() if hasattr(m, "convert_to_fp16") else None)
        self.latent_expander.apply(lambda m: m.convert_to_fp16() if hasattr(m, "convert_to_fp16") else None)
        self.decoder_vtx.apply(lambda m: m.convert_to_fp16() if hasattr(m, "convert_to_fp16") else None)
        self.decoder_vtx_ca.apply(lambda m: m.convert_to_fp16() if hasattr(m, "convert_to_fp16") else None)
        self.latent_proj.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.encoder.apply(lambda m: m.convert_to_fp32() if hasattr(m, "convert_to_fp32") else None)
        self.latent_expander.apply(lambda m: m.convert_to_fp32() if hasattr(m, "convert_to_fp32") else None)
        self.decoder_vtx.apply(lambda m: m.convert_to_fp32() if hasattr(m, "convert_to_fp32") else None)
        self.decoder_vtx_ca.apply(lambda m: m.convert_to_fp32() if hasattr(m, "convert_to_fp32") else None)
        self.latent_proj.apply(convert_module_to_f32)
