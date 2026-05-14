import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, max_alpha: float = 10.0):
        super().__init__()
        self.gamma = gamma
        self.max_alpha = max_alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos = targets.sum()
        neg = targets.numel() - pos
        alpha = torch.clamp(neg / (pos + 1e-6), max=self.max_alpha)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        return (alpha * (1 - pt) ** self.gamma * bce_loss).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = inputs.sum() + targets.sum()
        dice_score = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return 1.0 - dice_score


class AsymmetricFocalLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 1,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = False,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.targets = None
        self.anti_targets = None
        self.xs_pos = None
        self.xs_neg = None
        self.asymmetric_w = None
        self.loss = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.targets = y
        self.anti_targets = 1 - y
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(
                1 - self.xs_pos - self.xs_neg,
                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets,
            )
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        final_loss = -self.loss.sum()
        num_positives = y.sum()
        if num_positives == 0:
            return torch.tensor(0.0, device=x.device)
        return final_loss / num_positives


def load_pretrained_woself(
    checkpoint_path: str,
    vae: nn.Module,
    vertex_encoder: nn.Module | None = None,
    voxel_encoder: nn.Module | None = None,
    edge_encoder: nn.Module | None = None,
    query_decoder: nn.Module | None = None,
    active_encoder: nn.Module | None = None,
    connection_head: nn.Module | None = None,
    optimizer=None,
    ema_model=None,
):
    """Load compatible checkpoint weights into the provided model components."""
    if not os.path.exists(checkpoint_path):
        print(f"[INFO] Checkpoint not found at '{checkpoint_path}'. Models will start from scratch.")
        return {"epoch": 0, "best_loss": float("inf")}

    print(f"Loading pretrained models from: {os.path.basename(checkpoint_path)}")
    try:
        device = next(iter(vae.parameters())).device
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        print(f"[ERROR] Failed to load checkpoint {checkpoint_path}. Error: {exc}")
        return {"epoch": 0, "best_loss": float("inf")}

    def _load_and_log(model: nn.Module, model_name: str, state_dict):
        if model is None:
            return
        if state_dict is None:
            print(f"[WARN] No state found for '{model_name}' in checkpoint. It will remain initialized from scratch.")
            return

        current_state = model.state_dict()
        filtered_state = {}
        for key, value in state_dict.items():
            if key in current_state and value.shape == current_state[key].shape:
                filtered_state[key] = value
            else:
                expected_shape = tuple(current_state.get(key, torch.empty(0)).shape)
                print(f"[INFO] Skip loading {key}: checkpoint {tuple(value.shape)} != model {expected_shape}")

        missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
        print(f"--- Loading status for '{model_name}' ---")
        if not missing_keys and not unexpected_keys:
            print("Success: All weights loaded perfectly.")
            return
        if missing_keys:
            examples = ", ".join(missing_keys[:3]) + ("..." if len(missing_keys) > 3 else "")
            print(f"[INFO] {len(missing_keys)} keys missing: {examples}")
        if unexpected_keys:
            examples = ", ".join(unexpected_keys[:3]) + ("..." if len(unexpected_keys) > 3 else "")
            print(f"[WARN] {len(unexpected_keys)} unexpected keys: {examples}")

    _load_and_log(vae, "VoxelVAE", checkpoint.get("vae"))
    _load_and_log(query_decoder, "QueryPointDecoder", checkpoint.get("query_decoder"))
    _load_and_log(edge_encoder, "EdgeEncoder", checkpoint.get("edge_encoder"))
    _load_and_log(active_encoder, "ActiveEncoder", checkpoint.get("active_encoder"))
    _load_and_log(connection_head, "ConnectionHead", checkpoint.get("connection_head"))
    _load_and_log(voxel_encoder, "VoxelEncoder", checkpoint.get("voxel_encoder"))
    _load_and_log(vertex_encoder, "VertexEncoder", checkpoint.get("vtx_encoder"))

    if optimizer is not None and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("\n[INFO] Successfully loaded optimizer state.")
        except ValueError as exc:
            print(f"\n[WARN] Could not load optimizer state. It will be reset. Error: {exc}")

    if ema_model is not None:
        if "ema_state_dict" in checkpoint:
            try:
                ema_model.load_state_dict(checkpoint["ema_state_dict"])
                ema_model.to(device)
                print("\n[INFO] Successfully loaded EMA state.")
            except Exception as exc:
                print(f"\n[WARN] Failed to load EMA state. EMA will start fresh. Error: {exc}")
        else:
            print("\n[INFO] No 'ema_state_dict' found in checkpoint. EMA will start fresh.")

    original_epoch = checkpoint.get("epoch", "unknown")
    best_loss = checkpoint.get("best_loss", checkpoint.get("loss", float("inf")))
    print(f"\nSuccessfully processed checkpoint. Original epoch was {original_epoch}, best loss {best_loss:.4f}.")
    return {"epoch": checkpoint.get("epoch", 0), "best_loss": best_loss}


def fast_isin(target_coords: torch.Tensor, query_coords: torch.Tensor, resolution: int) -> torch.Tensor:
    """Vectorized membership test for integer 3D or batched 4D coordinates."""
    device = target_coords.device
    if target_coords.shape[-1] == 4:
        weight = torch.tensor([resolution ** 3, resolution ** 2, resolution, 1], device=device)
    elif target_coords.shape[-1] == 3:
        weight = torch.tensor([resolution ** 3, resolution ** 2, resolution], device=device)
    else:
        raise ValueError("Coordinates must have last dimension 3 or 4.")

    target_hash = (target_coords * weight).sum(dim=1)
    query_hash = (query_coords * weight).sum(dim=1)
    query_hash_sorted, _ = torch.sort(torch.unique(query_hash))

    idx = torch.searchsorted(query_hash_sorted, target_hash)
    in_bounds = idx < len(query_hash_sorted)
    matches = torch.zeros_like(target_hash, dtype=torch.bool)
    matches[in_bounds] = query_hash_sorted[idx[in_bounds]] == target_hash[in_bounds]
    return matches
