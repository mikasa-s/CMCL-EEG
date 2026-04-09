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


class PureInfoNCEPretrainLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.info_nce = SymmetricInfoNCELoss(temperature=temperature)

    def forward(self, eeg_shared: torch.Tensor, fmri_shared: torch.Tensor) -> dict[str, torch.Tensor]:
        contrastive = self.info_nce(eeg_shared, fmri_shared)
        zero = contrastive.new_zeros(())
        return {
            "loss": contrastive,
            "contrastive_loss": contrastive,
            "band_power_loss": zero,
            "separation_loss": zero,
        }


class DCCALoss(torch.nn.Module):
    def __init__(self, reg: float = 1e-4, eps: float = 1e-9) -> None:
        super().__init__()
        self.reg = float(reg)
        self.eps = float(eps)

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        if view1.ndim != 2 or view2.ndim != 2:
            raise ValueError("DCCA loss expects 2D feature tensors")
        if view1.shape != view2.shape:
            raise ValueError(f"DCCA views must have matching shapes, got {tuple(view1.shape)} and {tuple(view2.shape)}")

        batch_size, feat_dim = view1.shape
        if batch_size <= 1:
            return view1.new_zeros(())

        h1 = view1 - view1.mean(dim=0, keepdim=True)
        h2 = view2 - view2.mean(dim=0, keepdim=True)

        denom = float(max(batch_size - 1, 1))
        sigma11 = (h1.t() @ h1) / denom
        sigma22 = (h2.t() @ h2) / denom
        sigma12 = (h1.t() @ h2) / denom

        identity = torch.eye(feat_dim, device=view1.device, dtype=view1.dtype)
        sigma11 = sigma11 + self.reg * identity
        sigma22 = sigma22 + self.reg * identity

        evals_11, evecs_11 = torch.linalg.eigh(sigma11)
        evals_22, evecs_22 = torch.linalg.eigh(sigma22)
        evals_11 = torch.clamp(evals_11, min=self.eps)
        evals_22 = torch.clamp(evals_22, min=self.eps)

        sigma11_inv_sqrt = evecs_11 @ torch.diag(evals_11.rsqrt()) @ evecs_11.t()
        sigma22_inv_sqrt = evecs_22 @ torch.diag(evals_22.rsqrt()) @ evecs_22.t()

        t_matrix = sigma11_inv_sqrt @ sigma12 @ sigma22_inv_sqrt
        singular_values = torch.linalg.svdvals(t_matrix)
        correlation = singular_values.sum()
        return -correlation


class DCCAPretrainLoss(torch.nn.Module):
    def __init__(self, reg: float = 1e-4, eps: float = 1e-9) -> None:
        super().__init__()
        self.dcca = DCCALoss(reg=reg, eps=eps)

    def forward(self, eeg_shared: torch.Tensor, fmri_shared: torch.Tensor) -> dict[str, torch.Tensor]:
        correlation_loss = self.dcca(eeg_shared, fmri_shared)
        zero = correlation_loss.new_zeros(())
        return {
            "loss": correlation_loss,
            "contrastive_loss": correlation_loss,
            "band_power_loss": zero,
            "separation_loss": zero,
        }
