from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from tqdm import tqdm

BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 40.0),
)


def update_subject_pack_metadata(pack_dir: Path, array_name: str, array: np.ndarray) -> None:
    metadata_path = pack_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    else:
        metadata = {"format": "subject_memmap_v1", "arrays": {}}

    arrays_meta = metadata.setdefault("arrays", {})
    arrays_meta[array_name] = {
        "dtype": str(array.dtype),
        "shape": [int(dim) for dim in array.shape],
    }

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute 5D EEG band-power targets from already-preprocessed EEG segments and attach them to a prepared cache.")
    parser.add_argument("--manifest-csv", type=Path, required=True, help="Prepared manifest CSV produced after the original EEG preprocessing pipeline.")
    parser.add_argument("--root-dir", type=Path, default=None, help="Root directory for relative paths. Defaults to manifest parent.")
    parser.add_argument("--output-manifest", type=Path, default=None, help="Optional output manifest path. Defaults to overwrite input manifest for per-sample caches.")
    parser.add_argument("--sample-rate-hz", type=float, default=200.0, help="Sampling rate of the already-preprocessed EEG segments used for band-power estimation.")
    parser.add_argument("--window-sec", type=float, default=8.0, help="EEG segment duration in seconds after preprocessing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing band-power targets.")
    return parser.parse_args()


def resolve_path(root_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root_dir / path


def flatten_eeg_sample(eeg: np.ndarray) -> np.ndarray:
    array = np.asarray(eeg, dtype=np.float32)
    if array.ndim == 3:
        channels, seq_len, patch_len = array.shape
        return array.reshape(channels, seq_len * patch_len)
    if array.ndim == 2:
        return array
    raise ValueError(f"Unsupported EEG sample shape for band-power computation: {array.shape}")


def compute_band_power(eeg: np.ndarray, sample_rate_hz: float, expected_length: int | None = None) -> np.ndarray:
    flattened = flatten_eeg_sample(eeg)
    if expected_length is not None and flattened.shape[-1] != expected_length:
        raise ValueError(
            f"Expected a preprocessed EEG segment of length {expected_length}, got {flattened.shape[-1]}. "
            f"Please run the original preprocessing first and use its output for band-power extraction."
        )
    freqs, psd = welch(flattened, fs=sample_rate_hz, nperseg=flattened.shape[-1], axis=-1)
    band_powers: list[float] = []
    for _, low, high in BANDS:
        if high == BANDS[-1][2]:
            mask = (freqs >= low) & (freqs <= high)
        else:
            mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            band_powers.append(0.0)
            continue
        band_power = np.trapz(psd[:, mask], freqs[mask], axis=-1).mean()
        band_powers.append(float(band_power))
    return np.asarray(band_powers, dtype=np.float32)


def enrich_subject_packs(manifest: pd.DataFrame, root_dir: Path, sample_rate_hz: float, expected_length: int, overwrite: bool) -> None:
    for subject_rel_path in tqdm(manifest["subject_path"].dropna().unique().tolist(), desc="Band-power subject packs"):
        subject_dir = resolve_path(root_dir, str(subject_rel_path))
        eeg_path = subject_dir / "eeg.npy"
        band_power_path = subject_dir / "band_power.npy"
        if band_power_path.exists() and not overwrite:
            continue
        eeg = np.load(eeg_path, mmap_mode="r", allow_pickle=False)
        band_power = np.stack([compute_band_power(sample, sample_rate_hz, expected_length=expected_length) for sample in eeg], axis=0)
        del eeg
        np.save(band_power_path, band_power.astype(np.float32, copy=False))
        update_subject_pack_metadata(subject_dir, "band_power", band_power)


def enrich_sample_manifest(
    manifest: pd.DataFrame,
    root_dir: Path,
    output_manifest: Path,
    sample_rate_hz: float,
    expected_length: int,
    overwrite: bool,
) -> None:
    band_power_dir = output_manifest.parent / "band_power"
    band_power_dir.mkdir(parents=True, exist_ok=True)
    updated_rows: list[dict[str, object]] = []
    for row in tqdm(manifest.to_dict("records"), desc="Band-power samples"):
        updated = dict(row)
        sample_id = str(updated.get("sample_id", len(updated_rows)))
        relative_output = Path("band_power") / f"{sample_id}.npy"
        absolute_output = output_manifest.parent / relative_output
        if absolute_output.exists() and not overwrite:
            updated["band_power_path"] = relative_output.as_posix()
            updated_rows.append(updated)
            continue
        eeg_path = resolve_path(root_dir, str(updated["eeg_path"]))
        band_power = compute_band_power(
            np.load(eeg_path, mmap_mode="r", allow_pickle=False),
            sample_rate_hz,
            expected_length=expected_length,
        )
        np.save(absolute_output, band_power.astype(np.float32, copy=False))
        updated["band_power_path"] = relative_output.as_posix()
        updated_rows.append(updated)
    pd.DataFrame(updated_rows).to_csv(output_manifest, index=False)


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest_csv.resolve()
    root_dir = args.root_dir.resolve() if args.root_dir is not None else manifest_path.parent.resolve()
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    expected_length = int(round(float(args.sample_rate_hz) * float(args.window_sec)))
    print(
        f"Computing EEG band-power targets with sample_rate_hz={args.sample_rate_hz}, "
        f"window_sec={args.window_sec}, expected_length={expected_length}, bands={[name for name, _, _ in BANDS]}"
    )

    if "subject_path" in manifest.columns:
        enrich_subject_packs(
            manifest,
            root_dir=root_dir,
            sample_rate_hz=float(args.sample_rate_hz),
            expected_length=expected_length,
            overwrite=bool(args.overwrite),
        )
        if args.output_manifest and args.output_manifest.resolve() != manifest_path:
            manifest.to_csv(args.output_manifest, index=False)
        return

    output_manifest = args.output_manifest.resolve() if args.output_manifest is not None else manifest_path
    enrich_sample_manifest(
        manifest,
        root_dir=root_dir,
        output_manifest=output_manifest,
        sample_rate_hz=float(args.sample_rate_hz),
        expected_length=expected_length,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
