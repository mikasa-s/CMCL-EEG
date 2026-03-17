from __future__ import annotations

"""LaBraM adapter for EEG feature extraction."""

import csv
from pathlib import Path

import torch
import torch.nn as nn

from ..checkpoint_utils import load_compatible_state_dict

LABRAM_MAX_CHANNEL_SLOTS = 62
LABRAM_SHARED_CHANNEL_COUNT = 40
JOINT_CHANNEL_MANIFEST = Path(__file__).resolve().parents[2] / "cache" / "joint_contrastive" / "eeg_channels_target.csv"


def _normalize_channel_name(name: str) -> str:
    return str(name).strip().upper().replace(" ", "")


def _load_channel_names_from_manifest(manifest_path: str | Path) -> list[str]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"EEG channel manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        names: list[str] = []
        for row in reader:
            name = str(row.get("target_channel_name", "")).strip()
            if name:
                names.append(name)
    if not names:
        raise ValueError(f"No target_channel_name entries found in EEG channel manifest: {path}")
    return names


class EEGLaBraMAdapter(nn.Module):
    def __init__(
        self,
        model_name: str = "labram_base_patch200_200",
        checkpoint_path: str = "",
        freeze_backbone: bool = False,
        channel_manifest_path: str = "",
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
        self.expected_num_channels = LABRAM_MAX_CHANNEL_SLOTS
        self.dropped_channel_names: list[str] = []
        if str(channel_manifest_path).strip():
            current_names = _load_channel_names_from_manifest(channel_manifest_path)
            if JOINT_CHANNEL_MANIFEST.exists():
                joint_names = _load_channel_names_from_manifest(JOINT_CHANNEL_MANIFEST)
                joint_lookup = {_normalize_channel_name(name) for name in joint_names}
                leading = [name for name in joint_names if _normalize_channel_name(name) in {_normalize_channel_name(v) for v in current_names}]
                trailing = [name for name in current_names if _normalize_channel_name(name) not in joint_lookup]
                reordered_names = leading + trailing
            else:
                reordered_names = current_names
            if len(reordered_names) > self.expected_num_channels:
                self.dropped_channel_names = reordered_names[self.expected_num_channels:]

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

    def _prepare_input(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_input_channels = int(eeg.shape[1])
        if num_input_channels > self.expected_num_channels:
            eeg = eeg[:, : self.expected_num_channels, ...]
            num_input_channels = self.expected_num_channels
        preserved_channels = min(num_input_channels, LABRAM_SHARED_CHANNEL_COUNT)
        extra_channels = max(0, num_input_channels - preserved_channels)
        input_chans = [0]
        input_chans.extend(range(1, preserved_channels + 1))
        if extra_channels > 0:
            input_chans.extend(range(LABRAM_SHARED_CHANNEL_COUNT + 1, LABRAM_SHARED_CHANNEL_COUNT + extra_channels + 1))
        if num_input_channels == self.expected_num_channels:
            return eeg, None
        return eeg, torch.tensor(input_chans, dtype=torch.long, device=eeg.device)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 4:
            raise ValueError(f"LaBraM baseline expects EEG [B,C,S,P], got {tuple(eeg.shape)}")
        eeg, input_chans = self._prepare_input(eeg)
        features = self.backbone.forward_features(eeg, input_chans=input_chans)
        if features.ndim != 2:
            raise RuntimeError(f"Unexpected LaBraM feature shape: {tuple(features.shape)}")
        return features
