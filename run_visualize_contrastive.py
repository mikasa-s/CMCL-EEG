from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from mmcontrast.checkpoint_utils import extract_state_dict, load_checkpoint_file
from mmcontrast.config import TrainConfig
from mmcontrast.datasets import PairedEEGfMRIDataset
from mmcontrast.models import EEGfMRIContrastiveModel
from mmcontrast.visualization import save_cross_modal_similarity_heatmap, save_shared_private_tsne

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def next_indexed_output_path(output_dir: Path, stem: str, suffix: str) -> Path:
    pattern = f"{stem}_*{suffix}"
    max_index = 0
    for path in output_dir.glob(pattern):
        suffix_text = path.stem[len(stem) + 1 :]
        if suffix_text.isdigit():
            max_index = max(max_index, int(suffix_text))
    return output_dir / f"{stem}_{max_index + 1:03d}{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Offline visualization for contrastive EEG-fMRI representations")
    parser.add_argument("--config", type=str, default="configs/train_joint_contrastive.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Contrastive checkpoint path.")
    parser.add_argument("--manifest", type=str, default="", help="Optional manifest override.")
    parser.add_argument("--root-dir", type=str, default="", help="Optional dataset root override.")
    parser.add_argument("--output-dir", type=str, default="outputs/visualizations/contrastive", help="Directory for PNG/JSON outputs.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0, help="Maximum number of paired samples to run through the model. Use 0 or a negative value to run the full dataset.")
    parser.add_argument("--tsne-max-points", type=int, default=200, help="Maximum points per embedding group in the t-SNE plot.")
    parser.add_argument("--heatmap-max-points", type=int, default=48, help="Maximum paired samples shown in the similarity heatmap.")
    parser.add_argument("--device", type=str, default="", help="Explicit device, e.g. cpu or cuda:0.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Generic override in the form section.key=value. Can be repeated.",
    )
    return parser.parse_args()


def assign_nested_value(payload: dict[str, Any], dotted_key: str, value: object) -> None:
    parts = dotted_key.split(".")
    cursor: dict[str, Any] = payload
    for key in parts[:-1]:
        next_value = cursor.get(key)
        if next_value is None:
            next_value = {}
            cursor[key] = next_value
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot override nested key '{dotted_key}' because '{key}' is not a mapping")
        cursor = next_value
    cursor[parts[-1]] = value


def load_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if args.manifest.strip():
        assign_nested_value(config, "data.manifest_csv", args.manifest.strip())
    if args.root_dir.strip():
        assign_nested_value(config, "data.root_dir", args.root_dir.strip())
    for override in args.overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected section.key=value")
        dotted_key, raw_value = override.split("=", 1)
        assign_nested_value(config, dotted_key.strip(), yaml.safe_load(raw_value))
    return config


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_dataset(cfg: dict[str, Any]) -> PairedEEGfMRIDataset:
    data_cfg = dict(cfg["data"])
    manifest_path = str(data_cfg.get("manifest_csv", "")).strip()
    if not manifest_path:
        raise ValueError("data.manifest_csv must be configured for visualization")
    data_cfg["manifest_csv"] = str(resolve_path(manifest_path))
    if data_cfg.get("root_dir"):
        data_cfg["root_dir"] = str(resolve_path(str(data_cfg["root_dir"])))
    data_cfg.pop("train_manifest_csv", None)
    data_cfg.pop("val_manifest_csv", None)
    data_cfg.pop("test_manifest_csv", None)
    data_cfg.pop("expected_eeg_shape", None)
    data_cfg.pop("expected_fmri_shape", None)
    data_cfg["require_eeg"] = True
    data_cfg["require_fmri"] = True
    data_cfg["require_band_power"] = False
    return PairedEEGfMRIDataset(**data_cfg)


def build_model(cfg: dict[str, Any], device: torch.device) -> EEGfMRIContrastiveModel:
    model_cfg = {
        "train": dict(cfg["train"]),
        "data": dict(cfg["data"]),
        "eeg_model": dict(cfg["eeg_model"]),
        "fmri_model": dict(cfg["fmri_model"]),
    }
    model_cfg["eeg_model"]["checkpoint_path"] = ""
    model_cfg["fmri_model"]["checkpoint_path"] = ""
    model = EEGfMRIContrastiveModel(model_cfg).to(device)
    return model


def load_model_checkpoint(model: EEGfMRIContrastiveModel, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = load_checkpoint_file(str(checkpoint_path))
    state_dict = extract_state_dict(checkpoint, preferred_keys=("model", "module", "state_dict"))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {
        "missing_count": int(len(missing)),
        "unexpected_count": int(len(unexpected)),
        "missing": [str(item) for item in missing[:20]],
        "unexpected": [str(item) for item in unexpected[:20]],
        "device": str(device),
    }


def collect_embeddings(
    model: EEGfMRIContrastiveModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    outputs: dict[str, list[torch.Tensor]] = {
        "eeg_shared": [],
        "eeg_private": [],
        "fmri_shared": [],
    }
    model.eval()
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device, non_blocking=True)
            fmri = batch["fmri"].to(device, non_blocking=True)
            batch_out = model(eeg=eeg, fmri=fmri)
            outputs["eeg_shared"].append(batch_out["eeg_shared"].detach().cpu())
            outputs["eeg_private"].append(batch_out["eeg_private"].detach().cpu())
            outputs["fmri_shared"].append(batch_out["fmri_shared"].detach().cpu())
    collected = {key: torch.cat(value, dim=0) for key, value in outputs.items() if value}
    if not collected:
        raise RuntimeError("No embeddings were collected for visualization.")
    return collected


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args)
    TrainConfig(config).validate(base_dir=str(PROJECT_ROOT))

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = resolve_path(args.checkpoint)
    requested_device = args.device.strip()
    if requested_device:
        device = torch.device(requested_device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = build_dataset(config)
    if args.max_samples > 0 and len(dataset) > args.max_samples:
        dataset = Subset(dataset, list(range(args.max_samples)))
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=bool(args.num_workers > 0),
    )

    model = build_model(config, device=device)
    checkpoint_report = load_model_checkpoint(model, checkpoint_path, device=device)
    print(
        "Loaded contrastive checkpoint: "
        f"{checkpoint_path} "
        f"(missing={checkpoint_report['missing_count']}, unexpected={checkpoint_report['unexpected_count']})",
        flush=True,
    )

    embeddings = collect_embeddings(model, loader, device=device)
    tsne_path = next_indexed_output_path(output_dir, "tsne_shared_private", ".png")
    heatmap_path = next_indexed_output_path(output_dir, "cross_modal_similarity_heatmap", ".png")
    summary_path = next_indexed_output_path(output_dir, "visualization_summary", ".json")
    tsne_report = save_shared_private_tsne(
        embeddings["eeg_shared"],
        embeddings["eeg_private"],
        embeddings["fmri_shared"],
        output_path=tsne_path,
        max_points=max(3, int(args.tsne_max_points)),
    )
    heatmap_report = save_cross_modal_similarity_heatmap(
        embeddings["eeg_shared"],
        embeddings["fmri_shared"],
        output_path=heatmap_path,
        max_points=max(2, int(args.heatmap_max_points)),
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "num_samples": int(embeddings["eeg_shared"].shape[0]),
        "checkpoint_report": checkpoint_report,
        "tsne_report": tsne_report,
        "heatmap_report": heatmap_report,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Saved t-SNE: {tsne_path}", flush=True)
    print(f"Saved heatmap: {heatmap_path}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
