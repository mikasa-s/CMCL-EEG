from __future__ import annotations

"""LaBraM adapter for EEG feature extraction."""

import torch
import torch.nn as nn

from ..checkpoint_utils import load_compatible_state_dict


class EEGLaBraMAdapter(nn.Module):
    def __init__(
        self,
        model_name: str = "labram_base_patch200_200",
        checkpoint_path: str = "",
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        try:
            from ..backbones.eeg_labram.modeling_finetune import (
                labram_base_patch200_200,
                labram_huge_patch200_200,
                labram_large_patch200_200,
            )

            labram_factory = {
                "labram_base_patch200_200": labram_base_patch200_200,
                "labram_large_patch200_200": labram_large_patch200_200,
                "labram_huge_patch200_200": labram_huge_patch200_200,
            }
            if model_name not in labram_factory:
                raise ValueError(f"Unsupported LaBraM model_name: {model_name}")

            self.backbone = labram_factory[model_name](pretrained=False, num_classes=0)
        except ModuleNotFoundError as exc:
            if exc.name == "timm":
                raise ModuleNotFoundError("LaBraM baseline requires the 'timm' package. Please install timm>=0.9.16.") from exc
            raise

        self.feature_dim = int(getattr(self.backbone, "num_features", 200))

        if checkpoint_path:
            load_compatible_state_dict(
                self.backbone,
                checkpoint_path,
                preferred_keys=("model", "module", "state_dict"),
                prefixes=("module.", "model."),
            )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 4:
            raise ValueError(f"LaBraM baseline expects EEG [B,C,S,P], got {tuple(eeg.shape)}")
        if eeg.shape[1] != 62:
            raise ValueError(
                f"LaBraM baseline currently requires 62 EEG channels, but got {int(eeg.shape[1])}. "
                "Please remap/select channels to 62 before finetune."
            )

        features = self.backbone.forward_features(eeg)
        if features.ndim != 2:
            raise RuntimeError(f"Unexpected LaBraM feature shape: {tuple(features.shape)}")
        return features
