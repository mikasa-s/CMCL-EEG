from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from scipy.ndimage import zoom
from scipy.signal import resample
from tqdm import tqdm

from split_utils import write_loso_splits, write_subject_splits


TASK_LABELS = {
    "motorloc": 0,
    "MIpre": 1,
    "MIpost": 2,
    "eegNF": 3,
    "fmriNF": 4,
    "eegfmriNF": 5,
}

TASK_DURATIONS_SEC = {
    "motorloc": 320,
    "MIpre": 200,
    "MIpost": 200,
    "eegNF": 400,
    "fmriNF": 400,
    "eegfmriNF": 400,
}

TRIAL_TYPE_BINARY_LABELS = {
    "rest": 0,
    "task-me": 1,
    "task-mi": 1,
    "task-nf": 1,
}

TRIAL_TYPE_LABEL_NAMES = {
    0: "non_motor",
    1: "motor",
}


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    subject: str
    task: str
    trial_type: str
    eeg_path: str
    fmri_path: str
    label: int
    label_name: str
    eeg_shape: str
    fmri_shape: str


@dataclass(frozen=True)
class SubjectRecord:
    subject: str
    subject_path: str
    sample_count: int
    eeg_shape: str
    fmri_shape: str
    label_shape: str


@dataclass(frozen=True)
class SkippedBlockRecord:
    subject: str
    task: str
    block_index: int
    trial_type: str
    onset_sec: float
    duration_sec: float
    reason: str


@dataclass(frozen=True)
class MissingPairRecord:
    subject: str
    task: str
    eeg_path: str
    fmri_path: str
    reason: str


@dataclass(frozen=True)
class WindowPlacement:
    eeg_start_sec: float
    fmri_start_sec: float
    duration_sec: float
    protocol_onset_sec: float
    shift_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ds002336 EEG/fMRI pairs for this repository.")
    parser.add_argument("--ds-root", type=Path, required=True, help="Path to ds002336 root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output directory for converted arrays and manifests.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["motorloc", "MIpre", "MIpost", "eegNF", "fmriNF", "eegfmriNF"],
        choices=list(TASK_LABELS.keys()),
        help="Tasks to export.",
    )
    parser.add_argument(
        "--sample-mode",
        default="run",
        choices=["run", "block"],
        help="run exports one sample per run; block exports one sample per rest/task block.",
    )
    parser.add_argument(
        "--label-mode",
        default="task",
        choices=["task", "binary_rest_task"],
        help="task uses task categories as labels; binary_rest_task uses non_motor=0 and motor=1.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subject IDs such as sub-xp101. Defaults to all subjects found under ds-root.",
    )
    parser.add_argument(
        "--atlas-labels-img",
        type=Path,
        default=None,
        help="Optional path to a labels atlas image in MNI space. If omitted, Schaefer atlas is fetched.",
    )
    parser.add_argument("--atlas-name", default="schaefer", choices=["schaefer"], help="Built-in atlas preset.")
    parser.add_argument("--n-rois", type=int, default=400, help="ROI count for the built-in Schaefer atlas.")
    parser.add_argument(
        "--fmri-target-rois",
        type=int,
        default=None,
        help="Optional target ROI count for fMRI. Disabled by default because ROI interpolation changes the atlas semantics.",
    )
    parser.add_argument(
        "--fmri-mode",
        default="roi",
        choices=["roi", "volume"],
        help="roi saves [ROI,T]; volume saves normalized 4D windows [H,W,D,T].",
    )
    parser.add_argument(
        "--fmri-source",
        default="raw",
        choices=["raw", "spm_unsmoothed", "spm_smoothed"],
        help="raw reads original func/*_bold.nii.gz; spm_* reads derivatives/spm12_preproc outputs.",
    )
    parser.add_argument(
        "--fmri-preproc-root",
        type=Path,
        default=None,
        help="Optional root for SPM preprocessed fMRI. Defaults to <ds-root>/derivatives/spm12_preproc when --fmri-source is not raw.",
    )
    parser.add_argument("--tr", type=float, default=2.0, help="fMRI repetition time.")
    parser.add_argument(
        "--discard-initial-trs",
        type=int,
        default=1,
        help="Initial BOLD volumes to discard before any slicing. Defaults to 1 because ds002336 README states the scanner starts 2 seconds before protocol onset and TR=2s.",
    )
    parser.add_argument(
        "--protocol-offset-sec",
        type=float,
        default=2.0,
        help="Seconds to subtract from task events TSV onsets to map them onto protocol time. Defaults to 2.0 for ds002336.",
    )
    parser.add_argument(
        "--fmri-target-t",
        type=int,
        default=None,
        help="Optional target time length for fMRI. Disabled by default because temporal interpolation changes the raw TR grid.",
    )
    parser.add_argument(
        "--fmri-voxel-size",
        nargs=3,
        type=float,
        default=[2.0, 2.0, 2.0],
        help="Target spatial voxel size in mm for volume-mode fMRI preprocessing.",
    )
    parser.add_argument(
        "--fmri-max-shape",
        nargs=3,
        type=int,
        default=[48, 48, 48],
        help="Maximum center-cropped spatial shape after volume resampling.",
    )
    parser.add_argument(
        "--fmri-float16",
        action="store_true",
        help="Save preprocessed volume-mode fMRI windows as float16 instead of float32.",
    )
    parser.add_argument(
        "--allow-fmri-roi-resample",
        action="store_true",
        help="Explicitly allow Fourier resampling on the ROI axis. Use only for debugging, not for raw-faithful exports.",
    )
    parser.add_argument(
        "--allow-fmri-time-resample",
        action="store_true",
        help="Explicitly allow Fourier resampling on the time axis. Use only for debugging, not for raw-faithful exports.",
    )
    parser.add_argument(
        "--eeg-mode",
        default="continuous",
        choices=["continuous", "patched"],
        help="continuous saves [C,T]; patched saves [C,S,P] for direct use by the default dataset.",
    )
    parser.add_argument(
        "--eeg-seq-len",
        type=int,
        default=None,
        help="Optional EEG patch count when eeg-mode=patched. Defaults to 30 for run mode and to block duration in seconds for block mode.",
    )
    parser.add_argument(
        "--eeg-patch-len",
        type=int,
        default=None,
        help="Optional EEG patch length when eeg-mode=patched. Defaults to the EEG sampling rate so each patch spans about one second.",
    )
    parser.add_argument(
        "--drop-ecg",
        action="store_true",
        help="Drop ECG and other non-EEG channels when reading the preprocessed BrainVision files.",
    )
    parser.add_argument(
        "--standardize-fmri",
        action="store_true",
        help="Apply standardization inside the ROI masker.",
    )
    parser.add_argument(
        "--pack-subject-files",
        action="store_true",
        help="Pack all exported samples of the same subject into one NPZ file, so downstream loading is subject-packed.",
    )
    parser.add_argument(
        "--split-mode",
        default="loso",
        choices=["none", "subject", "loso"],
        help="Optional split generation after preprocessing.",
    )
    parser.add_argument(
        "--split-output-dir",
        type=Path,
        default=None,
        help="Optional directory for generated split manifests. Defaults to <output-root>/splits_subjectwise or <output-root>/loso_subjectwise.",
    )
    parser.add_argument("--train-subjects", type=int, default=7, help="Number of subjects in train split when --split-mode=subject.")
    parser.add_argument("--val-subjects", type=int, default=2, help="Number of subjects in val split or LOSO validation subjects.")
    parser.add_argument("--test-subjects", type=int, default=1, help="Number of subjects in test split when --split-mode=subject.")
    return parser.parse_args()


def find_subjects(ds_root: Path, requested_subjects: list[str] | None) -> list[str]:
    if requested_subjects:
        return requested_subjects
    return sorted(path.name for path in ds_root.glob("sub-*") if path.is_dir())


def resolve_fmri_path(ds_root: Path, subject: str, task: str, args: argparse.Namespace) -> Path:
    if args.fmri_source == "raw":
        return ds_root / subject / "func" / f"{subject}_task-{task}_bold.nii.gz"

    fmri_preproc_root = args.fmri_preproc_root.resolve() if args.fmri_preproc_root is not None else (ds_root / "derivatives" / "spm12_preproc")
    subject_dir = fmri_preproc_root / subject
    if args.fmri_source == "spm_smoothed":
        flat_final = subject_dir / f"{subject}_task-{task}_bold.nii"
        if flat_final.exists():
            return flat_final
        legacy_task_dir = subject_dir / f"task-{task}"
        legacy_final = legacy_task_dir / "fmri_final.nii"
        if legacy_final.exists():
            return legacy_final
        return legacy_task_dir / f"swratrim_{subject}_task-{task}_bold.nii"

    legacy_task_dir = subject_dir / f"task-{task}"
    legacy_unsmoothed = legacy_task_dir / f"wratrim_{subject}_task-{task}_bold.nii"
    return legacy_unsmoothed


def get_atlas_labels_img(args: argparse.Namespace, atlas_cache_dir: Path) -> str:
    if args.atlas_labels_img is not None:
        return str(args.atlas_labels_img)
    atlas_cache_dir.mkdir(parents=True, exist_ok=True)
    atlas = fetch_atlas_schaefer_2018(
        n_rois=args.n_rois,
        yeo_networks=7,
        resolution_mm=2,
        data_dir=str(atlas_cache_dir),
    )
    return str(atlas.maps)


def _normalize_marker_description(description: str) -> str:
    return str(description).strip().upper().replace(" ", "")


def detect_eeg_protocol_start_sec(raw: mne.io.BaseRaw) -> float:
    first_s99: float | None = None
    first_s2: float | None = None
    for onset, description in zip(raw.annotations.onset, raw.annotations.description):
        marker = _normalize_marker_description(description)
        if marker.endswith("/S99") and first_s99 is None:
            first_s99 = float(onset)
        if marker.endswith("/S2") and first_s2 is None:
            first_s2 = float(onset)
    if first_s99 is not None:
        return first_s99
    if first_s2 is not None:
        return max(0.0, first_s2 - 20.0)
    raise ValueError("Could not locate EEG protocol start marker S99 or fallback S2 in BrainVision annotations")


def load_eeg(eeg_vhdr_path: Path, drop_ecg: bool) -> tuple[np.ndarray, float, float]:
    raw = mne.io.read_raw_brainvision(str(eeg_vhdr_path), preload=True, verbose="ERROR")
    protocol_start_sec = detect_eeg_protocol_start_sec(raw)
    if drop_ecg:
        raw = raw.pick("eeg")
    data = raw.get_data().astype(np.float32)
    return data, float(raw.info["sfreq"]), float(protocol_start_sec)


def crop_eeg_to_task(data: np.ndarray, task: str, sfreq: float) -> np.ndarray:
    duration_sec = TASK_DURATIONS_SEC[task]
    target_samples = int(round(duration_sec * sfreq))
    if data.shape[1] < target_samples:
        raise ValueError(f"EEG samples shorter than expected for task {task}: {data.shape[1]} < {target_samples}")
    return data[:, :target_samples]


def crop_eeg_to_duration(data: np.ndarray, sfreq: float, duration_sec: float) -> np.ndarray:
    target_samples = int(round(duration_sec * sfreq))
    if data.shape[1] < target_samples:
        raise ValueError(f"EEG samples shorter than requested duration: {data.shape[1]} < {target_samples}")
    return data[:, :target_samples]


def maybe_patch_eeg(data: np.ndarray, seq_len: int, patch_len: int) -> np.ndarray:
    target_len = seq_len * patch_len
    if data.shape[1] != target_len:
        data = resample(data, target_len, axis=1)
    return data.reshape(data.shape[0], seq_len, patch_len).astype(np.float32)


def resolve_eeg_patch_params(
    sfreq: float,
    requested_seq_len: int | None,
    requested_patch_len: int | None,
    sample_mode: str,
    duration_sec: float | None = None,
) -> tuple[int, int]:
    patch_len = requested_patch_len if requested_patch_len is not None else int(round(sfreq))
    if patch_len <= 0:
        raise ValueError(f"EEG patch length must be positive, got {patch_len}")

    if requested_seq_len is not None:
        seq_len = requested_seq_len
    elif sample_mode == "block":
        if duration_sec is None:
            raise ValueError("Block-mode EEG patch inference requires duration_sec.")
        seq_len = max(1, int(round(duration_sec)))
    else:
        seq_len = 30

    if seq_len <= 0:
        raise ValueError(f"EEG sequence length must be positive, got {seq_len}")
    return seq_len, patch_len


def resample_fmri_if_needed(
    series: np.ndarray,
    fmri_target_rois: int | None,
    fmri_target_t: int | None,
    allow_roi_resample: bool,
    allow_time_resample: bool,
) -> np.ndarray:
    if fmri_target_rois is not None and series.shape[0] != fmri_target_rois:
        if not allow_roi_resample:
            raise ValueError(
                f"Requested fmri_target_rois={fmri_target_rois}, but extracted ROI count is {series.shape[0]}. "
                "Provide a real atlas with the desired ROI count instead of interpolating, or pass --allow-fmri-roi-resample to override."
            )
        series = resample(series, fmri_target_rois, axis=0).astype(np.float32)
    if fmri_target_t is not None and series.shape[1] != fmri_target_t:
        if not allow_time_resample:
            raise ValueError(
                f"Requested fmri_target_t={fmri_target_t}, but the block contains {series.shape[1]} time points. "
                "Keep the native TR grid for raw-faithful exports, or pass --allow-fmri-time-resample to override."
            )
        series = resample(series, fmri_target_t, axis=1).astype(np.float32)
    return series.astype(np.float32)


def extract_roi_timeseries(
    fmri_nii_path: Path,
    labels_img: str,
    tr: float,
    discard_initial_trs: int,
    standardize_fmri: bool,
    fmri_target_t: int | None,
    allow_time_resample: bool,
) -> np.ndarray:
    img = nib.load(str(fmri_nii_path))
    if discard_initial_trs > 0:
        img = nib.Nifti1Image(img.get_fdata(dtype=np.float32)[..., discard_initial_trs:], img.affine, img.header)
    masker = NiftiLabelsMasker(labels_img=labels_img, standardize=standardize_fmri, detrend=True, t_r=tr)
    series = masker.fit_transform(img)
    series = series.T.astype(np.float32)
    return resample_fmri_if_needed(series, None, fmri_target_t, allow_roi_resample=False, allow_time_resample=allow_time_resample)


def load_bold_volume(fmri_nii_path: Path, discard_initial_trs: int) -> tuple[np.ndarray, tuple[float, ...]]:
    img = nib.load(str(fmri_nii_path))
    volume = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    if discard_initial_trs > 0:
        volume = volume[..., discard_initial_trs:]
    return volume, tuple(float(dim) for dim in img.header.get_zooms())


def spatial_resample_volume(data: np.ndarray, voxel_size: tuple[float, ...], target_voxel_size: tuple[float, ...]) -> np.ndarray:
    if data.ndim != 4:
        raise ValueError(f"Expected 4D BOLD volume [H,W,D,T], got {data.shape}")
    scale_factors = tuple(float(current) / float(target) for current, target in zip(voxel_size[:3], target_voxel_size[:3]))
    return zoom(data, zoom=(scale_factors[0], scale_factors[1], scale_factors[2], 1.0), order=1).astype(np.float32)


def temporal_resample_volume(data: np.ndarray, source_tr: float, target_tr: float) -> np.ndarray:
    if data.ndim != 4:
        raise ValueError(f"Expected 4D BOLD volume [H,W,D,T], got {data.shape}")
    if abs(source_tr - target_tr) < 1e-6:
        return data.astype(np.float32)
    target_t = max(1, int(round(data.shape[3] * float(source_tr) / float(target_tr))))
    return zoom(data, zoom=(1.0, 1.0, 1.0, target_t / max(data.shape[3], 1)), order=1).astype(np.float32)


def center_crop_spatial_max(data: np.ndarray, max_shape: tuple[int, ...]) -> np.ndarray:
    if data.ndim != 4:
        raise ValueError(f"Expected 4D BOLD volume [H,W,D,T], got {data.shape}")
    slices: list[slice] = []
    for axis, max_size in enumerate(max_shape[:3]):
        current_size = data.shape[axis]
        if current_size > int(max_size):
            start = (current_size - int(max_size)) // 2
            end = start + int(max_size)
            slices.append(slice(start, end))
        else:
            slices.append(slice(0, current_size))
    slices.append(slice(None))
    return data[tuple(slices)].astype(np.float32)


def preprocess_fmri_volume(
    data: np.ndarray,
    voxel_size: tuple[float, ...],
    source_tr: float,
    args: argparse.Namespace,
) -> np.ndarray:
    output = spatial_resample_volume(data, voxel_size=voxel_size, target_voxel_size=tuple(args.fmri_voxel_size))
    output = temporal_resample_volume(output, source_tr=source_tr, target_tr=args.tr)
    output = center_crop_spatial_max(output, max_shape=tuple(args.fmri_max_shape))
    if args.fmri_float16:
        return output.astype(np.float16)
    return output.astype(np.float32)


def load_task_events(ds_root: Path, task: str) -> pd.DataFrame:
    events_path = ds_root / f"task-{task}_events.tsv"
    if not events_path.exists():
        raise FileNotFoundError(f"Task events TSV not found: {events_path}")
    rows: list[dict[str, object]] = []
    with open(events_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Events TSV is empty: {events_path}")
        index_by_name = {name: idx for idx, name in enumerate(header)}
        required_columns = ["onset", "duration", "trial_type"]
        missing = [name for name in required_columns if name not in index_by_name]
        if missing:
            raise ValueError(f"Events TSV missing required columns {missing}: {events_path}")
        for parts in reader:
            if not parts or not any(cell.strip() for cell in parts):
                continue
            onset = parts[index_by_name["onset"]].strip() if index_by_name["onset"] < len(parts) else ""
            duration = parts[index_by_name["duration"]].strip() if index_by_name["duration"] < len(parts) else ""
            trial_type = parts[index_by_name["trial_type"]].strip() if index_by_name["trial_type"] < len(parts) else ""
            if not onset or not duration or not trial_type:
                continue
            rows.append(
                {
                    "onset": float(onset),
                    "duration": float(duration),
                    "trial_type": trial_type,
                }
            )
    if not rows:
        raise ValueError(f"No valid event rows found in {events_path}")
    return pd.DataFrame(rows)


def slice_eeg_block(data: np.ndarray, sfreq: float, start_sec: float, duration_sec: float) -> np.ndarray:
    start = int(round(start_sec * sfreq))
    length = int(round(duration_sec * sfreq))
    end = start + length
    if start < 0 or end > data.shape[1]:
        raise ValueError(f"EEG block slice out of range: start={start}, end={end}, total={data.shape[1]}")
    return data[:, start:end].astype(np.float32)


def slice_fmri_block(series: np.ndarray, tr: float, start_sec: float, duration_sec: float) -> np.ndarray:
    start = int(round(start_sec / tr))
    length = int(round(duration_sec / tr))
    end = start + length
    if start < 0 or end > series.shape[1]:
        raise ValueError(f"fMRI block slice out of range: start={start}, end={end}, total={series.shape[1]}")
    return series[:, start:end].astype(np.float32)


def slice_fmri_volume_block(volume: np.ndarray, tr: float, start_sec: float, duration_sec: float) -> np.ndarray:
    start = int(round(start_sec / tr))
    length = int(round(duration_sec / tr))
    end = start + length
    if start < 0 or end > volume.shape[3]:
        raise ValueError(f"fMRI volume block slice out of range: start={start}, end={end}, total={volume.shape[3]}")
    return volume[:, :, :, start:end].astype(np.float32)


def compute_shifted_window(
    eeg_total_sec: float,
    eeg_protocol_start_sec: float,
    fmri_total_sec: float,
    protocol_onset_sec: float,
    duration_sec: float,
) -> WindowPlacement | None:
    eeg_start_sec = eeg_protocol_start_sec + protocol_onset_sec
    fmri_start_sec = protocol_onset_sec

    if eeg_start_sec < 0 or fmri_start_sec < 0:
        return None

    eeg_overflow = max(0.0, eeg_start_sec + duration_sec - eeg_total_sec)
    fmri_overflow = max(0.0, fmri_start_sec + duration_sec - fmri_total_sec)
    shift_sec = max(eeg_overflow, fmri_overflow)

    if shift_sec > 0:
        eeg_start_sec -= shift_sec
        fmri_start_sec -= shift_sec
        protocol_onset_sec -= shift_sec

    if eeg_start_sec < 0 or fmri_start_sec < 0:
        return None
    if eeg_start_sec + duration_sec > eeg_total_sec + 1e-6:
        return None
    if fmri_start_sec + duration_sec > fmri_total_sec + 1e-6:
        return None

    return WindowPlacement(
        eeg_start_sec=float(eeg_start_sec),
        fmri_start_sec=float(fmri_start_sec),
        duration_sec=float(duration_sec),
        protocol_onset_sec=float(protocol_onset_sec),
        shift_sec=float(shift_sec),
    )


def block_fits_eeg(data: np.ndarray, sfreq: float, start_sec: float, duration_sec: float) -> bool:
    start = int(round(start_sec * sfreq))
    length = int(round(duration_sec * sfreq))
    end = start + length
    return start >= 0 and end <= data.shape[1]


def block_fits_fmri_matrix(series: np.ndarray, tr: float, start_sec: float, duration_sec: float) -> bool:
    start = int(round(start_sec / tr))
    length = int(round(duration_sec / tr))
    end = start + length
    return start >= 0 and end <= series.shape[1]


def block_fits_fmri_volume(volume: np.ndarray, tr: float, start_sec: float, duration_sec: float) -> bool:
    start = int(round(start_sec / tr))
    length = int(round(duration_sec / tr))
    end = start + length
    return start >= 0 and end <= volume.shape[3]


def resolve_binary_label(trial_type: str) -> tuple[int, str]:
    normalized = trial_type.strip().lower()
    if normalized not in TRIAL_TYPE_BINARY_LABELS:
        raise ValueError(f"Unsupported trial_type for binary labeling: {trial_type}")
    label = TRIAL_TYPE_BINARY_LABELS[normalized]
    return label, TRIAL_TYPE_LABEL_NAMES[label]


def build_sample_record(
    sample_id: str,
    subject: str,
    task: str,
    trial_type: str,
    eeg_rel_path: Path,
    fmri_rel_path: Path,
    label: int,
    label_name: str,
    eeg: np.ndarray,
    fmri: np.ndarray,
) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        subject=subject,
        task=task,
        trial_type=trial_type,
        eeg_path=eeg_rel_path.as_posix(),
        fmri_path=fmri_rel_path.as_posix(),
        label=label,
        label_name=label_name,
        eeg_shape="x".join(str(dim) for dim in eeg.shape),
        fmri_shape="x".join(str(dim) for dim in fmri.shape),
    )


def iter_subject_task_pairs(subjects: Iterable[str], tasks: Iterable[str]) -> Iterable[tuple[str, str]]:
    for subject in subjects:
        for task in tasks:
            yield subject, task


def stack_subject_samples(samples: list[np.ndarray], name: str) -> np.ndarray:
    if not samples:
        raise ValueError(f"No {name} samples available for subject packing.")
    first_shape = tuple(samples[0].shape)
    for sample in samples[1:]:
        if tuple(sample.shape) != first_shape:
            raise ValueError(f"Cannot pack subject-level {name} arrays with inconsistent shapes: {first_shape} vs {tuple(sample.shape)}")
    return np.stack(samples, axis=0)


def main() -> None:
    args = parse_args()
    if args.label_mode == "binary_rest_task" and args.sample_mode != "block":
        raise ValueError("binary_rest_task requires --sample-mode block because each run contains both rest and task blocks.")
    if args.fmri_mode == "volume" and args.fmri_target_rois is not None:
        raise ValueError("--fmri-target-rois is only valid when --fmri-mode=roi")

    ds_root = args.ds_root.resolve()
    out_root = args.output_root.resolve()
    eeg_out_dir = out_root / "eeg"
    fmri_out_dir = out_root / "fmri"
    packed_out_dir = out_root / "subjects"
    atlas_cache_dir = out_root / "atlas_cache"
    out_root.mkdir(parents=True, exist_ok=True)
    if args.pack_subject_files:
        packed_out_dir.mkdir(parents=True, exist_ok=True)
    else:
        eeg_out_dir.mkdir(parents=True, exist_ok=True)
        fmri_out_dir.mkdir(parents=True, exist_ok=True)

    labels_img = get_atlas_labels_img(args, atlas_cache_dir) if args.fmri_mode == "roi" else ""
    subjects = find_subjects(ds_root, args.subjects)
    fmri_discard_initial_trs = int(args.discard_initial_trs)
    protocol_offset_sec = float(args.protocol_offset_sec)
    if args.fmri_source != "raw":
        if fmri_discard_initial_trs != 0 or abs(protocol_offset_sec) > 1e-6:
            print(
                "Using SPM-preprocessed fMRI: overriding discard_initial_trs to 0 and protocol_offset_sec to 0.0 "
                "because the first lead-in TR has already been removed in the SPM pipeline."
            )
        fmri_discard_initial_trs = 0
        protocol_offset_sec = 0.0
        if args.fmri_mode == "volume" and tuple(args.fmri_max_shape) != (79, 95, 79):
            print(
                f"Warning: SPM-preprocessed MNI volumes are typically about 79x95x79, but fmri_max_shape={tuple(args.fmri_max_shape)}. "
                "Further cropping may remove brain coverage before training."
            )

    records: list[SampleRecord] = []
    subject_records: list[SubjectRecord] = []
    skipped_blocks: list[SkippedBlockRecord] = []
    missing_pairs: list[MissingPairRecord] = []
    for subject in tqdm(subjects, desc="Preparing ds002336"):
        packed_eeg_samples: list[np.ndarray] = []
        packed_fmri_samples: list[np.ndarray] = []
        packed_labels: list[int] = []
        packed_sample_ids: list[str] = []
        packed_tasks: list[str] = []
        packed_trial_types: list[str] = []

        for task in args.tasks:
            eeg_vhdr = ds_root / "derivatives" / subject / "eeg_pp" / f"{subject}_task-{task}_eeg_pp.vhdr"
            fmri_nii = resolve_fmri_path(ds_root, subject, task, args)
            missing_reasons: list[str] = []
            if not eeg_vhdr.exists():
                missing_reasons.append("missing_eeg")
            if not fmri_nii.exists():
                missing_reasons.append("missing_fmri")
            if missing_reasons:
                missing_pairs.append(
                    MissingPairRecord(
                        subject=subject,
                        task=task,
                        eeg_path=str(eeg_vhdr),
                        fmri_path=str(fmri_nii),
                        reason="+".join(missing_reasons),
                    )
                )
                continue

            eeg, sfreq, eeg_protocol_start_sec = load_eeg(eeg_vhdr, drop_ecg=args.drop_ecg)
            if args.sample_mode == "run":
                eeg = crop_eeg_to_task(eeg, task, sfreq=sfreq)
                if args.eeg_mode == "patched":
                    eeg_seq_len, eeg_patch_len = resolve_eeg_patch_params(
                        sfreq=sfreq,
                        requested_seq_len=args.eeg_seq_len,
                        requested_patch_len=args.eeg_patch_len,
                        sample_mode=args.sample_mode,
                    )
                    eeg = maybe_patch_eeg(eeg, seq_len=eeg_seq_len, patch_len=eeg_patch_len)

                if args.fmri_mode == "roi":
                    fmri = extract_roi_timeseries(
                        fmri_nii_path=fmri_nii,
                        labels_img=labels_img,
                        tr=args.tr,
                        discard_initial_trs=fmri_discard_initial_trs,
                        standardize_fmri=args.standardize_fmri,
                        fmri_target_t=args.fmri_target_t,
                        allow_time_resample=args.allow_fmri_time_resample,
                    )
                    fmri = resample_fmri_if_needed(
                        fmri,
                        args.fmri_target_rois,
                        None,
                        allow_roi_resample=args.allow_fmri_roi_resample,
                        allow_time_resample=args.allow_fmri_time_resample,
                    )
                else:
                    fmri_volume, voxel_size = load_bold_volume(
                        fmri_nii_path=fmri_nii,
                        discard_initial_trs=fmri_discard_initial_trs,
                    )
                    fmri = preprocess_fmri_volume(
                        fmri_volume,
                        voxel_size=voxel_size,
                        source_tr=float(voxel_size[3]) if len(voxel_size) > 3 else float(args.tr),
                        args=args,
                    )

                sample_id = f"{subject}_{task}"
                if args.pack_subject_files:
                    packed_eeg_samples.append(eeg.astype(np.float32))
                    packed_fmri_samples.append(fmri.astype(np.float32))
                    packed_labels.append(int(TASK_LABELS[task]))
                    packed_sample_ids.append(sample_id)
                    packed_tasks.append(task)
                    packed_trial_types.append(task)
                else:
                    eeg_out_path = eeg_out_dir / f"{sample_id}.npy"
                    fmri_out_path = fmri_out_dir / f"{sample_id}.npy"
                    np.save(eeg_out_path, eeg)
                    np.save(fmri_out_path, fmri)

                    records.append(
                        build_sample_record(
                            sample_id=sample_id,
                            subject=subject,
                            task=task,
                            trial_type=task,
                            eeg_rel_path=eeg_out_path.relative_to(out_root),
                            fmri_rel_path=fmri_out_path.relative_to(out_root),
                            label=TASK_LABELS[task],
                            label_name=task,
                            eeg=eeg,
                            fmri=fmri,
                        )
                    )
                continue

            events = load_task_events(ds_root, task)
            if args.fmri_mode == "roi":
                fmri_full = extract_roi_timeseries(
                    fmri_nii_path=fmri_nii,
                    labels_img=labels_img,
                    tr=args.tr,
                    discard_initial_trs=fmri_discard_initial_trs,
                    standardize_fmri=args.standardize_fmri,
                    fmri_target_t=None,
                    allow_time_resample=args.allow_fmri_time_resample,
                )
            else:
                fmri_volume, voxel_size = load_bold_volume(
                    fmri_nii_path=fmri_nii,
                    discard_initial_trs=fmri_discard_initial_trs,
                )
                fmri_full = preprocess_fmri_volume(
                    fmri_volume,
                    voxel_size=voxel_size,
                    source_tr=float(voxel_size[3]) if len(voxel_size) > 3 else float(args.tr),
                    args=args,
                )

            eeg_total_sec = float(eeg.shape[1]) / float(sfreq)
            fmri_total_sec = float(fmri_full.shape[1]) * float(args.tr) if args.fmri_mode == "roi" else float(fmri_full.shape[3]) * float(args.tr)

            for block_idx, row in events.reset_index(drop=True).iterrows():
                onset_sec = float(row["onset"]) - protocol_offset_sec
                duration_sec = float(row["duration"])
                trial_type = str(row["trial_type"]).strip()
                placement = compute_shifted_window(
                    eeg_total_sec=eeg_total_sec,
                    eeg_protocol_start_sec=eeg_protocol_start_sec,
                    fmri_total_sec=fmri_total_sec,
                    protocol_onset_sec=onset_sec,
                    duration_sec=duration_sec,
                )
                if placement is None:
                    skipped_blocks.append(
                        SkippedBlockRecord(
                            subject=subject,
                            task=task,
                            block_index=int(block_idx),
                            trial_type=trial_type,
                            onset_sec=float(onset_sec),
                            duration_sec=float(duration_sec),
                            reason="paired_window_out_of_range",
                        )
                    )
                    continue
                eeg_block = slice_eeg_block(eeg, sfreq=sfreq, start_sec=placement.eeg_start_sec, duration_sec=duration_sec)
                if args.eeg_mode == "patched":
                    eeg_seq_len, eeg_patch_len = resolve_eeg_patch_params(
                        sfreq=sfreq,
                        requested_seq_len=args.eeg_seq_len,
                        requested_patch_len=args.eeg_patch_len,
                        sample_mode=args.sample_mode,
                        duration_sec=duration_sec,
                    )
                    eeg_block = maybe_patch_eeg(eeg_block, seq_len=eeg_seq_len, patch_len=eeg_patch_len)

                if args.fmri_mode == "roi":
                    fmri_block = slice_fmri_block(fmri_full, tr=args.tr, start_sec=placement.fmri_start_sec, duration_sec=duration_sec)
                    fmri_block = resample_fmri_if_needed(
                        fmri_block,
                        args.fmri_target_rois,
                        args.fmri_target_t,
                        allow_roi_resample=args.allow_fmri_roi_resample,
                        allow_time_resample=args.allow_fmri_time_resample,
                    )
                else:
                    fmri_block = slice_fmri_volume_block(fmri_full, tr=args.tr, start_sec=placement.fmri_start_sec, duration_sec=duration_sec)

                if args.label_mode == "binary_rest_task":
                    label, label_name = resolve_binary_label(trial_type)
                else:
                    label = TASK_LABELS[task]
                    label_name = task

                sample_id = f"{subject}_{task}_block-{block_idx:02d}"
                if args.pack_subject_files:
                    packed_eeg_samples.append(eeg_block.astype(np.float32))
                    packed_fmri_samples.append(fmri_block.astype(np.float32))
                    packed_labels.append(int(label))
                    packed_sample_ids.append(sample_id)
                    packed_tasks.append(task)
                    packed_trial_types.append(trial_type)
                else:
                    eeg_out_path = eeg_out_dir / f"{sample_id}.npy"
                    fmri_out_path = fmri_out_dir / f"{sample_id}.npy"
                    np.save(eeg_out_path, eeg_block)
                    np.save(fmri_out_path, fmri_block)

                    records.append(
                        build_sample_record(
                            sample_id=sample_id,
                            subject=subject,
                            task=task,
                            trial_type=trial_type,
                            eeg_rel_path=eeg_out_path.relative_to(out_root),
                            fmri_rel_path=fmri_out_path.relative_to(out_root),
                            label=label,
                            label_name=label_name,
                            eeg=eeg_block,
                            fmri=fmri_block,
                        )
                    )

        if args.pack_subject_files and packed_eeg_samples:
            packed_eeg = stack_subject_samples(packed_eeg_samples, name="EEG")
            packed_fmri = stack_subject_samples(packed_fmri_samples, name="fMRI")
            packed_labels_array = np.asarray(packed_labels, dtype=np.int64)
            packed_sample_ids_array = np.asarray(packed_sample_ids)
            packed_tasks_array = np.asarray(packed_tasks)
            packed_trial_types_array = np.asarray(packed_trial_types)

            subject_path = packed_out_dir / f"{subject}.npz"
            np.savez(
                subject_path,
                eeg=packed_eeg,
                fmri=packed_fmri,
                labels=packed_labels_array,
                sample_id=packed_sample_ids_array,
                task=packed_tasks_array,
                trial_type=packed_trial_types_array,
            )
            subject_records.append(
                SubjectRecord(
                    subject=subject,
                    subject_path=subject_path.relative_to(out_root).as_posix(),
                    sample_count=int(packed_labels_array.shape[0]),
                    eeg_shape="x".join(str(dim) for dim in packed_eeg.shape),
                    fmri_shape="x".join(str(dim) for dim in packed_fmri.shape),
                    label_shape="x".join(str(dim) for dim in packed_labels_array.shape),
                )
            )

    if not records and not subject_records:
        raise RuntimeError("No samples were exported. Check subject IDs, task names, and input paths.")

    if subject_records:
        pd.DataFrame(record.__dict__ for record in subject_records).to_csv(out_root / "manifest_all.csv", index=False)
    else:
        pd.DataFrame(record.__dict__ for record in records).to_csv(out_root / "manifest_all.csv", index=False)
    if skipped_blocks:
        pd.DataFrame(record.__dict__ for record in skipped_blocks).to_csv(out_root / "skipped_blocks.csv", index=False)
    if missing_pairs:
        pd.DataFrame(record.__dict__ for record in missing_pairs).to_csv(out_root / "missing_pairs.csv", index=False)
        print(f"Skipped {len(missing_pairs)} subject-task pairs because EEG/fMRI files were not both present.")

    if args.split_mode == "subject":
        split_dir = args.split_output_dir.resolve() if args.split_output_dir else (out_root / "splits_subjectwise")
        write_subject_splits(
            manifest_path=out_root / "manifest_all.csv",
            output_dir=split_dir,
            train_subjects=int(args.train_subjects),
            val_subjects=int(args.val_subjects),
            test_subjects=int(args.test_subjects),
        )
    elif args.split_mode == "loso":
        split_dir = args.split_output_dir.resolve() if args.split_output_dir else (out_root / "loso_subjectwise")
        write_loso_splits(
            manifest_path=out_root / "manifest_all.csv",
            output_dir=split_dir,
            val_subjects=int(args.val_subjects),
        )


if __name__ == "__main__":
    main()
