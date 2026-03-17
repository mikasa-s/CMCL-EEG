from __future__ import annotations

import inspect
import torch
import torch.nn as nn

from .eeg_cbramod_adapter import EEGCBraModAdapter
from .fmri_adapter import FMRINeuroSTORMAdapter


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EEGSharedPrivateEncoder(nn.Module):
    def __init__(self, eeg_cfg: dict, head_cfg: dict | None = None) -> None:
        super().__init__()
        head_cfg = head_cfg or {}
        valid_backbone_keys = inspect.signature(EEGCBraModAdapter.__init__).parameters
        backbone_cfg = {key: value for key, value in eeg_cfg.items() if key in valid_backbone_keys}
        self.backbone = EEGCBraModAdapter(**backbone_cfg)
        backbone_dim = int(self.backbone.feature_dim)
        self.shared_dim = int(eeg_cfg.get("shared_dim", head_cfg.get("shared_dim", 256)))
        self.private_dim = int(eeg_cfg.get("private_dim", head_cfg.get("private_dim", self.shared_dim)))
        self.band_power_dim = int(eeg_cfg.get("band_power_dim", head_cfg.get("band_power_dim", 5)))
        head_dropout = float(head_cfg.get("head_dropout", 0.0))

        self.shared_head = LinearHead(backbone_dim, self.shared_dim, dropout=head_dropout)
        self.private_head = LinearHead(backbone_dim, self.private_dim, dropout=head_dropout)
        self.band_power_head = LinearHead(self.private_dim, self.band_power_dim, dropout=head_dropout)
        self.feature_dim = self.shared_dim + self.private_dim

    def forward(self, eeg: torch.Tensor) -> dict[str, torch.Tensor]:
        eeg_feat = self.backbone(eeg)
        eeg_shared = self.shared_head(eeg_feat)
        eeg_private = self.private_head(eeg_feat)
        band_power_pred = self.band_power_head(eeg_private)
        return {
            "eeg_feat": eeg_feat,
            "eeg_shared": eeg_shared,
            "eeg_private": eeg_private,
            "band_power_pred": band_power_pred,
        }

    def encode_for_finetune(self, eeg: torch.Tensor, mode: str = "concat") -> torch.Tensor:
        outputs = self.forward(eeg)
        normalized_mode = str(mode).strip().lower()
        if normalized_mode == "shared":
            return outputs["eeg_shared"]
        if normalized_mode == "private":
            return outputs["eeg_private"]
        if normalized_mode == "concat":
            return torch.cat((outputs["eeg_shared"], outputs["eeg_private"]), dim=-1)
        raise ValueError(f"Unsupported EEG classifier mode: {mode}")


class FMRISharedEncoder(nn.Module):
    def __init__(self, fmri_cfg: dict, head_cfg: dict | None = None) -> None:
        super().__init__()
        head_cfg = head_cfg or {}
        valid_backbone_keys = inspect.signature(FMRINeuroSTORMAdapter.__init__).parameters
        backbone_cfg = {key: value for key, value in fmri_cfg.items() if key in valid_backbone_keys}
        self.backbone = FMRINeuroSTORMAdapter(**backbone_cfg)
        backbone_dim = int(self.backbone.feature_dim)
        self.shared_dim = int(fmri_cfg.get("shared_dim", head_cfg.get("shared_dim", 256)))
        head_dropout = float(head_cfg.get("head_dropout", 0.0))
        self.shared_head = LinearHead(backbone_dim, self.shared_dim, dropout=head_dropout)
        self.feature_dim = self.shared_dim

    def forward(self, fmri: torch.Tensor) -> dict[str, torch.Tensor]:
        fmri_feat = self.backbone(fmri)
        fmri_shared = self.shared_head(fmri_feat)
        return {
            "fmri_feat": fmri_feat,
            "fmri_shared": fmri_shared,
        }
