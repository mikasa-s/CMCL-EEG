from __future__ import annotations

"""对比学习损失函数。"""

import torch
import torch.nn.functional as F

from .distributed import gather_with_grad, get_rank


class SymmetricInfoNCELoss(torch.nn.Module):
    """对 EEG->fMRI 和 fMRI->EEG 两个方向同时计算 InfoNCE。"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, eeg_embed: torch.Tensor, fmri_embed: torch.Tensor) -> torch.Tensor:
        """先跨卡聚合，再计算双向匹配损失。"""
        eeg_global = gather_with_grad(eeg_embed)
        fmri_global = gather_with_grad(fmri_embed)

        # 每一行都和全局 batch 做相似度，正确样本位于相同索引位置。
        logits_eeg = eeg_embed @ fmri_global.t() / self.temperature
        logits_fmri = fmri_embed @ eeg_global.t() / self.temperature

        batch = eeg_embed.size(0)
        rank = get_rank()
        labels = torch.arange(batch, device=eeg_embed.device) + rank * batch

        loss_eeg = F.cross_entropy(logits_eeg, labels)
        loss_fmri = F.cross_entropy(logits_fmri, labels)
        return 0.5 * (loss_eeg + loss_fmri)


def separation_cosine_loss(shared: torch.Tensor, private: torch.Tensor) -> torch.Tensor:
    shared_norm = F.normalize(shared, dim=-1)
    private_norm = F.normalize(private, dim=-1)
    cosine = (shared_norm * private_norm).sum(dim=-1)
    return (cosine ** 2).mean()


class SharedPrivatePretrainLoss(torch.nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        band_power_weight: float = 1.0,
        separation_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.info_nce = SymmetricInfoNCELoss(temperature=temperature)
        self.band_power_weight = float(band_power_weight)
        self.separation_weight = float(separation_weight)

    def forward(
        self,
        eeg_shared: torch.Tensor,
        fmri_shared: torch.Tensor,
        eeg_private: torch.Tensor,
        band_power_pred: torch.Tensor,
        band_power_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        contrastive = self.info_nce(eeg_shared, fmri_shared)
        band_power = F.mse_loss(band_power_pred, band_power_target)
        separation = separation_cosine_loss(eeg_shared, eeg_private)
        total = contrastive + (self.band_power_weight * band_power) + (self.separation_weight * separation)
        return {
            "loss": total,
            "contrastive_loss": contrastive,
            "band_power_loss": band_power,
            "separation_loss": separation,
        }
