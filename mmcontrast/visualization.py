from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_numpy(x: torch.Tensor, max_items: int | None = None) -> np.ndarray:
    if max_items is not None and x.shape[0] > max_items:
        x = x[:max_items]
    return x.detach().float().cpu().numpy()


def save_shared_private_tsne(
    eeg_shared: torch.Tensor,
    eeg_private: torch.Tensor,
    fmri_shared: torch.Tensor,
    output_path: str | Path,
    max_points: int = 200,
    random_state: int = 42,
) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Embedding visualization requires matplotlib and scikit-learn."
        ) from exc

    eeg_shared_np = _to_numpy(eeg_shared, max_items=max_points)
    eeg_private_np = _to_numpy(eeg_private, max_items=max_points)
    fmri_shared_np = _to_numpy(fmri_shared, max_items=max_points)

    labels = (
        ["EEG shared"] * len(eeg_shared_np)
        + ["EEG private"] * len(eeg_private_np)
        + ["fMRI shared"] * len(fmri_shared_np)
    )
    features = np.concatenate([eeg_shared_np, eeg_private_np, fmri_shared_np], axis=0)
    if features.shape[0] < 3:
        return {"saved": False, "reason": "not_enough_points"}

    perplexity = min(30, max(2, features.shape[0] // 6))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=random_state)
    coords = tsne.fit_transform(features)

    output = Path(output_path)
    _ensure_parent(output)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    color_map = {
        "EEG shared": "#1f77b4",
        "EEG private": "#d62728",
        "fMRI shared": "#2ca02c",
    }
    for label in ["EEG shared", "EEG private", "fMRI shared"]:
        mask = np.array([item == label for item in labels], dtype=bool)
        ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.75, label=label, c=color_map[label])
    ax.set_title("t-SNE of EEG/FMRI Shared-Private Representations")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {
        "saved": True,
        "path": str(output),
        "num_points_per_group": {
            "eeg_shared": int(len(eeg_shared_np)),
            "eeg_private": int(len(eeg_private_np)),
            "fmri_shared": int(len(fmri_shared_np)),
        },
    }


def save_cross_modal_similarity_heatmap(
    eeg_shared: torch.Tensor,
    fmri_shared: torch.Tensor,
    output_path: str | Path,
    max_points: int = 48,
) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Similarity heatmap visualization requires matplotlib."
        ) from exc

    eeg = eeg_shared[:max_points].detach().float().cpu()
    fmri = fmri_shared[:max_points].detach().float().cpu()
    if eeg.shape[0] == 0 or fmri.shape[0] == 0:
        return {"saved": False, "reason": "empty_embeddings"}

    eeg = torch.nn.functional.normalize(eeg, dim=-1)
    fmri = torch.nn.functional.normalize(fmri, dim=-1)
    sim = eeg @ fmri.t()
    sim_np = sim.numpy()

    output = Path(output_path)
    _ensure_parent(output)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    im = ax.imshow(sim_np, cmap="coolwarm", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_title("Cross-Modal Similarity Heatmap (EEG shared vs fMRI shared)")
    ax.set_xlabel("fMRI sample index")
    ax.set_ylabel("EEG sample index")
    diag_count = min(sim_np.shape[0], sim_np.shape[1])
    ax.plot(np.arange(diag_count), np.arange(diag_count), color="black", linewidth=0.8, linestyle="--")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    diagonal_mean = float(np.diag(sim_np[:diag_count, :diag_count]).mean()) if diag_count > 0 else float("nan")
    off_diag_mask = ~np.eye(sim_np.shape[0], sim_np.shape[1], dtype=bool)
    off_diagonal_mean = float(sim_np[off_diag_mask].mean()) if np.any(off_diag_mask) else float("nan")
    return {
        "saved": True,
        "path": str(output),
        "num_points": int(sim_np.shape[0]),
        "diagonal_mean": diagonal_mean,
        "off_diagonal_mean": off_diagonal_mean,
    }
