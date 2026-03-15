from __future__ import annotations

import argparse
import csv
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd
from scipy.signal import resample
from tqdm import tqdm

from preprocess_common import (
    add_atlas_args,
    build_canonical_subject_map,
    add_common_fmri_args,
    add_dataset_io_args,
    add_eeg_patch_args,
    build_channel_name_index,
    add_fmri_roi_resample_args,
    add_subject_args,
    add_subject_packing_and_split_args,
    add_training_ready_arg,
    extract_roi_timeseries,
    find_subjects,
    get_atlas_labels_img,
    load_target_channel_names,
    load_bold_volume,
    make_channel_metadata_rows,
    make_subject_uid,
    preprocess_fmri_volume,
    prepare_training_ready_eeg,
    prepare_training_ready_fmri,
    reorder_eeg_channels,
    resample_fmri_if_needed,
    stack_subject_samples,
    write_channel_metadata,
    write_subject_memmap_pack,
    write_subject_mapping,
    write_loso_splits,
    write_subject_splits,
)


TASK_LABELS = {
    "motorloc": 0,
    "MIpre": 1,
    "MIpost": 2,
    "eegNF": 3,
    "fmriNF": 4,
    "eegfmriNF": 5,
    "1dNF": 6,
    "2dNF": 7,
}

TASK_DURATIONS_SEC = {
    "motorloc": 320,
    "MIpre": 200,
    "MIpost": 200,
    "eegNF": 400,
    "fmriNF": 400,
    "eegfmriNF": 400,
    "1dNF": 320,
    "2dNF": 320,
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
    dataset: str
    subject: str
    subject_uid: str
    original_subject: str
    task: str
    trial_type: str
    eeg_path: str
    fmri_path: str
    label: int
    label_name: str
    eeg_shape: str
    fmri_shape: str
    training_ready: bool = False


@dataclass(frozen=True)
class SubjectRecord:
    dataset: str
    subject: str
    subject_uid: str
    original_subject: str
    subject_path: str
    sample_count: int
    eeg_shape: str
    fmri_shape: str
    label_shape: str
    training_ready: bool = False


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


@dataclass(frozen=True)
class TaskRunRecording:
    task_key: str
    task_name: str
    run: str
    eeg_vhdr: Path
    fmri_nii: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ds00233x (ds002336/ds002338) EEG/fMRI pairs for this repository.")
    add_dataset_io_args(parser, ds_root_help="Path to ds00233x root.", output_root_help="Output directory for converted arrays and manifests.")
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
    add_subject_args(parser, subject_example="sub-xp101")
    add_atlas_args(parser)
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
    add_common_fmri_args(
        parser,
        default_fmri_mode="roi",
        tr_help="fMRI repetition time.",
        standardize_help="Apply standardization inside the ROI masker.",
        fmri_max_shape_help="Maximum center-cropped spatial shape after volume resampling.",
    )
    parser.add_argument(
        "--discard-initial-trs",
        type=int,
        default=1,
        help="Initial BOLD volumes to discard before any slicing. Defaults to 1 because ds00233x README states the scanner starts 2 seconds before protocol onset and TR=2s.",
    )
    parser.add_argument(
        "--protocol-offset-sec",
        type=float,
        default=2.0,
        help="Seconds to subtract from task events TSV onsets to map them onto protocol time. Defaults to 2.0 for ds00233x.",
    )
    parser.add_argument(
        "--block-window-sec",
        type=float,
        default=8.0,
        help="When sample-mode=block, split each block into fixed windows of this length in seconds. If a block is shorter, use the full block duration.",
    )
    parser.add_argument(
        "--block-overlap-sec",
        type=float,
        default=2.0,
        help="When sample-mode=block, adjacent block windows overlap by this many seconds.",
    )
    add_fmri_roi_resample_args(parser)
    add_eeg_patch_args(
        parser,
        default_eeg_mode="continuous",
        default_seq_len=None,
        default_patch_len=None,
        seq_len_help="Optional EEG patch count when eeg-mode=patched. Defaults to 30 for run mode and to block duration in seconds for block mode.",
        patch_len_help="Optional EEG patch length when eeg-mode=patched. Defaults to the EEG sampling rate so each patch spans about one second.",
    )
    parser.add_argument(
        "--drop-ecg",
        action="store_true",
        help="Drop ECG and other non-EEG channels when reading the preprocessed BrainVision files.",
    )
    parser.add_argument(
        "--target-channel-manifest",
        type=Path,
        default=None,
        help="Optional channel manifest from joint pretraining. When provided, EEG channels are remapped to that exact normalized channel order.",
    )
    add_subject_packing_and_split_args(
        parser,
        pack_help="Pack all exported samples of the same subject into one directory of memmap-friendly NPY files, so downstream loading is subject-packed.",
        split_help="Optional split generation after preprocessing.",
        train_subjects=7,
        val_subjects=2,
        test_subjects=1,
    )
    add_training_ready_arg(parser)
    parser.add_argument("--eeg-only", action="store_true", help="Export EEG-only finetune cache. fMRI arrays are not saved.")
    parser.add_argument("--dataset-name", default="ds00233x", help="Dataset name written to manifests and sample IDs.")
    return parser.parse_args()


def resolve_fmri_path(ds_root: Path, subject: str, task: str, args: argparse.Namespace, run: str = "") -> Path:
    run_suffix = f"_{run}" if run else ""
    if args.fmri_source == "raw":
        return ds_root / subject / "func" / f"{subject}_task-{task}{run_suffix}_bold.nii.gz"

    fmri_preproc_root = args.fmri_preproc_root.resolve() if args.fmri_preproc_root is not None else (ds_root / "derivatives" / "spm12_preproc")
    subject_dir = fmri_preproc_root / subject
    if args.fmri_source == "spm_smoothed":
        flat_final = subject_dir / f"{subject}_task-{task}{run_suffix}_bold.nii"
        if flat_final.exists():
            return flat_final
        legacy_task_dir = subject_dir / f"task-{task}{run_suffix}"
        legacy_final = legacy_task_dir / "fmri_final.nii"
        if legacy_final.exists():
            return legacy_final
        return legacy_task_dir / f"swratrim_{subject}_task-{task}{run_suffix}_bold.nii"

    legacy_task_dir = subject_dir / f"task-{task}{run_suffix}"
    legacy_unsmoothed = legacy_task_dir / f"wratrim_{subject}_task-{task}{run_suffix}_bold.nii"
    return legacy_unsmoothed


def extract_run_token(path: Path) -> str:
    match = re.search(r"(_run-[^_]+)_eeg_pp\\.vhdr$", path.name)
    return match.group(1).lstrip("_") if match else ""


def discover_task_recordings(ds_root: Path, subject: str, task: str, args: argparse.Namespace) -> list[TaskRunRecording]:
    eeg_pp_dir = ds_root / "derivatives" / subject / "eeg_pp"
    candidates = sorted(eeg_pp_dir.glob(f"*{subject}_task-{task}*_eeg_pp.vhdr"))
    if not candidates:
        return []

    recordings: list[TaskRunRecording] = []
    seen: set[tuple[str, str, str]] = set()
    for eeg_vhdr in candidates:
        run = extract_run_token(eeg_vhdr)
        task_name = f"{task}_{run}" if run else task
        fmri_nii = resolve_fmri_path(ds_root, subject, task, args, run=run)
        key = (task, task_name, run)
        if key in seen:
            continue
        seen.add(key)
        recordings.append(TaskRunRecording(task_key=task, task_name=task_name, run=run, eeg_vhdr=eeg_vhdr, fmri_nii=fmri_nii))
    return recordings


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


def read_brainvision_eeg_only(eeg_vhdr_path: Path, preload: bool) -> mne.io.BaseRaw:
    # Some BrainVision files contain non-EEG channels (e.g., ECG) that can trigger
    # montage warnings in certain MNE versions/environments. We load and then keep EEG only.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Consider setting the channel types to be of EEG/sEEG/ECoG/DBS/fNIRS using inst.set_channel_types before calling inst.set_montage.*",
        )
        raw = mne.io.read_raw_brainvision(str(eeg_vhdr_path), preload=preload, verbose="ERROR")
    raw = raw.pick("eeg")
    return raw


def load_eeg(eeg_vhdr_path: Path, drop_ecg: bool) -> tuple[np.ndarray, float, float, list[str]]:
    raw = read_brainvision_eeg_only(eeg_vhdr_path, preload=True)
    protocol_start_sec = detect_eeg_protocol_start_sec(raw)
    data = raw.get_data().astype(np.float32)
    return data, float(raw.info["sfreq"]), float(protocol_start_sec), list(raw.ch_names)


def load_eeg_channel_names(eeg_vhdr_path: Path, drop_ecg: bool) -> list[str]:
    raw = read_brainvision_eeg_only(eeg_vhdr_path, preload=False)
    return list(raw.ch_names)


def compute_common_eeg_channels(ds_root: Path, subjects: list[str], tasks: list[str], drop_ecg: bool) -> list[str]:
    common_channels: set[str] | None = None
    ordered_channels: list[str] = []
    file_count = 0
    args_stub = argparse.Namespace(fmri_source="raw", fmri_preproc_root=None)
    for subject in subjects:
        for task in tasks:
            recordings = discover_task_recordings(ds_root, subject, task, args_stub)
            for recording in recordings:
                channel_names = load_eeg_channel_names(recording.eeg_vhdr, drop_ecg=drop_ecg)
                normalized_names = list(build_channel_name_index(channel_names).keys())
                if common_channels is None:
                    common_channels = set(normalized_names)
                    ordered_channels = normalized_names
                else:
                    common_channels &= set(normalized_names)
                file_count += 1

    if file_count == 0:
        raise RuntimeError("No EEG files found while computing ds00233x channel intersection.")
    if common_channels is None or not common_channels:
        raise RuntimeError("No shared EEG channels remain for ds00233x after normalization.")
    return [channel for channel in ordered_channels if channel in common_channels]


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


def build_block_subwindow_offsets(block_duration_sec: float, window_sec: float, overlap_sec: float) -> list[float]:
    """Return subwindow offsets (seconds) within one block using sliding windows.

    Examples with window=8 and overlap=2:
    - duration 20 -> [0, 6, 12]
    - duration 8  -> [0]
    - duration 5  -> [0]
    """
    if block_duration_sec <= 0:
        return []
    if block_duration_sec <= window_sec:
        return [0.0]

    stride_sec = window_sec - overlap_sec
    starts: list[float] = []
    start_sec = 0.0
    # Add regular stride windows.
    while start_sec + window_sec <= block_duration_sec + 1e-8:
        starts.append(float(start_sec))
        start_sec += stride_sec

    # Ensure the last window reaches the end of the block.
    last_start = max(0.0, block_duration_sec - window_sec)
    if not starts or abs(starts[-1] - last_start) > 1e-6:
        starts.append(float(last_start))
    return starts


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
    dataset: str,
    subject: str,
    subject_uid: str,
    original_subject: str,
    task: str,
    trial_type: str,
    eeg_rel_path: Path,
    fmri_rel_path: Path | None,
    label: int,
    label_name: str,
    eeg: np.ndarray,
    fmri: np.ndarray | None,
    training_ready: bool,
) -> SampleRecord:
    fmri_path_value = fmri_rel_path.as_posix() if fmri_rel_path is not None else ""
    fmri_shape_value = "x".join(str(dim) for dim in fmri.shape) if fmri is not None else ""
    return SampleRecord(
        sample_id=sample_id,
        dataset=dataset,
        subject=subject,
        subject_uid=subject_uid,
        original_subject=original_subject,
        task=task,
        trial_type=trial_type,
        eeg_path=eeg_rel_path.as_posix(),
        fmri_path=fmri_path_value,
        label=label,
        label_name=label_name,
        eeg_shape="x".join(str(dim) for dim in eeg.shape),
        fmri_shape=fmri_shape_value,
        training_ready=bool(training_ready),
    )


def iter_subject_task_pairs(subjects: Iterable[str], tasks: Iterable[str]) -> Iterable[tuple[str, str]]:
    for subject in subjects:
        for task in tasks:
            yield subject, task


def main() -> None:
    args = parse_args()
    if args.label_mode == "binary_rest_task" and args.sample_mode != "block":
        raise ValueError("binary_rest_task requires --sample-mode block because each run contains both rest and task blocks.")
    if args.fmri_mode == "volume" and args.fmri_target_rois is not None:
        raise ValueError("--fmri-target-rois is only valid when --fmri-mode=roi")
    if args.sample_mode == "block":
        if args.block_window_sec <= 0:
            raise ValueError("--block-window-sec must be positive")
        if args.block_overlap_sec < 0:
            raise ValueError("--block-overlap-sec must be non-negative")
        if args.block_overlap_sec >= args.block_window_sec:
            raise ValueError("--block-overlap-sec must be smaller than --block-window-sec")

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
        if not args.eeg_only:
            fmri_out_dir.mkdir(parents=True, exist_ok=True)

    labels_img = get_atlas_labels_img(args.atlas_labels_img, atlas_cache_dir, args.n_rois) if args.fmri_mode == "roi" else ""
    subjects = find_subjects(ds_root, args.subjects)
    canonical_subject_map = build_canonical_subject_map(subjects)
    target_channel_names = load_target_channel_names(args.target_channel_manifest.resolve()) if args.target_channel_manifest is not None else None
    if target_channel_names is None:
        target_channel_names = compute_common_eeg_channels(ds_root, subjects, list(args.tasks), drop_ecg=bool(args.drop_ecg))
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
    subject_mapping_rows: list[dict[str, object]] = []
    dataset_channel_rows: list[dict[str, object]] = []
    channel_mapping_rows: list[dict[str, object]] = []
    for subject in tqdm(subjects, desc="Preparing ds00233x"):
        canonical_subject = canonical_subject_map[subject]
        subject_uid = make_subject_uid(args.dataset_name, canonical_subject)
        subject_mapping_rows.append(
            {
                "dataset": args.dataset_name,
                "original_subject": subject,
                "subject": canonical_subject,
                "subject_uid": subject_uid,
            }
        )
        packed_eeg_samples: list[np.ndarray] = []
        packed_fmri_samples: list[np.ndarray] = []
        packed_labels: list[int] = []
        packed_sample_ids: list[str] = []
        packed_tasks: list[str] = []
        packed_trial_types: list[str] = []

        for task in args.tasks:
            recordings = discover_task_recordings(ds_root, subject, task, args)
            if not recordings:
                missing_pairs.append(
                    MissingPairRecord(
                        subject=subject,
                        task=task,
                        eeg_path="",
                        fmri_path="",
                        reason="missing_task_recordings",
                    )
                )
                continue
            for recording in recordings:
                eeg_vhdr = recording.eeg_vhdr
                fmri_nii = recording.fmri_nii
                task_name = recording.task_name
                missing_reasons: list[str] = []
                if not eeg_vhdr.exists():
                    missing_reasons.append("missing_eeg")
                if (not args.eeg_only) and (not fmri_nii.exists()):
                    missing_reasons.append("missing_fmri")
                if missing_reasons:
                    missing_pairs.append(
                        MissingPairRecord(
                            subject=subject,
                            task=task_name,
                            eeg_path=str(eeg_vhdr),
                            fmri_path=str(fmri_nii),
                            reason="+".join(missing_reasons),
                        )
                    )
                    continue

                eeg, sfreq, eeg_protocol_start_sec, eeg_channel_names = load_eeg(eeg_vhdr, drop_ecg=args.drop_ecg)
                if not dataset_channel_rows:
                    dataset_channel_rows.extend(make_channel_metadata_rows(args.dataset_name, eeg_channel_names))
                eeg, current_channel_mapping = reorder_eeg_channels(eeg, eeg_channel_names, target_channel_names)
                if not channel_mapping_rows:
                    for row in current_channel_mapping:
                        channel_mapping_rows.append({"dataset": args.dataset_name, **row})
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
                    eeg = prepare_training_ready_eeg(eeg, enabled=bool(args.training_ready))

                    fmri: np.ndarray | None = None
                    if not args.eeg_only:
                        if args.fmri_mode == "roi":
                            fmri = extract_roi_timeseries(
                                fmri_nii_path=fmri_nii,
                                labels_img=labels_img,
                                tr=args.tr,
                                standardize_fmri=args.standardize_fmri,
                                discard_initial_trs=fmri_discard_initial_trs,
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
                                target_voxel_size=tuple(args.fmri_voxel_size),
                                target_tr=float(args.tr),
                                max_shape=tuple(args.fmri_max_shape),
                                use_float16=bool(args.fmri_float16),
                            )
                        fmri = prepare_training_ready_fmri(fmri, fmri_mode=args.fmri_mode, enabled=bool(args.training_ready))

                    sample_id = f"{args.dataset_name}_{canonical_subject}_{task_name}"
                    if args.pack_subject_files:
                        packed_eeg_samples.append(eeg.astype(np.float32))
                        if fmri is not None:
                            packed_fmri_samples.append(fmri.astype(np.float32))
                        packed_labels.append(int(TASK_LABELS[task]))
                        packed_sample_ids.append(sample_id)
                        packed_tasks.append(task_name)
                        packed_trial_types.append(task_name)
                    else:
                        eeg_out_path = eeg_out_dir / f"{sample_id}.npy"
                        np.save(eeg_out_path, eeg)
                        fmri_out_path: Path | None = None
                        if fmri is not None:
                            fmri_out_path = fmri_out_dir / f"{sample_id}.npy"
                            np.save(fmri_out_path, fmri)

                        records.append(
                            build_sample_record(
                                sample_id=sample_id,
                                dataset=args.dataset_name,
                                subject=canonical_subject,
                                subject_uid=subject_uid,
                                original_subject=subject,
                                task=task_name,
                                trial_type=task_name,
                                eeg_rel_path=eeg_out_path.relative_to(out_root),
                                fmri_rel_path=None if fmri_out_path is None else fmri_out_path.relative_to(out_root),
                                label=TASK_LABELS[task],
                                label_name=task,
                                eeg=eeg,
                                fmri=fmri,
                                training_ready=bool(args.training_ready),
                            )
                        )
                    continue

                events = load_task_events(ds_root, task)
                fmri_full: np.ndarray | None = None
                if not args.eeg_only:
                    if args.fmri_mode == "roi":
                        fmri_full = extract_roi_timeseries(
                            fmri_nii_path=fmri_nii,
                            labels_img=labels_img,
                            tr=args.tr,
                            standardize_fmri=args.standardize_fmri,
                            discard_initial_trs=fmri_discard_initial_trs,
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
                            target_voxel_size=tuple(args.fmri_voxel_size),
                            target_tr=float(args.tr),
                            max_shape=tuple(args.fmri_max_shape),
                            use_float16=bool(args.fmri_float16),
                        )

                eeg_total_sec = float(eeg.shape[1]) / float(sfreq)
                fmri_total_sec = TASK_DURATIONS_SEC[task] if fmri_full is None else (float(fmri_full.shape[1]) * float(args.tr) if args.fmri_mode == "roi" else float(fmri_full.shape[3]) * float(args.tr))

                for block_idx, row in events.reset_index(drop=True).iterrows():
                    onset_sec = float(row["onset"]) - protocol_offset_sec
                    duration_sec = float(row["duration"])
                    trial_type = str(row["trial_type"]).strip()
                    block_window_sec = min(float(args.block_window_sec), float(duration_sec))
                    subwindow_offsets = build_block_subwindow_offsets(
                        block_duration_sec=float(duration_sec),
                        window_sec=float(block_window_sec),
                        overlap_sec=float(args.block_overlap_sec),
                    )
                    for sub_idx, sub_offset_sec in enumerate(subwindow_offsets):
                        sub_onset_sec = onset_sec + float(sub_offset_sec)
                        placement = compute_shifted_window(
                            eeg_total_sec=eeg_total_sec,
                            eeg_protocol_start_sec=eeg_protocol_start_sec,
                            fmri_total_sec=fmri_total_sec,
                            protocol_onset_sec=sub_onset_sec,
                            duration_sec=float(block_window_sec),
                        )
                        if placement is None:
                            skipped_blocks.append(
                                SkippedBlockRecord(
                                    subject=subject,
                                    task=task,
                                    block_index=int(block_idx),
                                    trial_type=trial_type,
                                    onset_sec=float(sub_onset_sec),
                                    duration_sec=float(block_window_sec),
                                    reason="paired_window_out_of_range",
                                )
                            )
                            continue
                        eeg_block = slice_eeg_block(eeg, sfreq=sfreq, start_sec=placement.eeg_start_sec, duration_sec=float(block_window_sec))
                        if args.eeg_mode == "patched":
                            eeg_seq_len, eeg_patch_len = resolve_eeg_patch_params(
                                sfreq=sfreq,
                                requested_seq_len=args.eeg_seq_len,
                                requested_patch_len=args.eeg_patch_len,
                                sample_mode=args.sample_mode,
                                duration_sec=float(block_window_sec),
                            )
                            eeg_block = maybe_patch_eeg(eeg_block, seq_len=eeg_seq_len, patch_len=eeg_patch_len)
                        eeg_block = prepare_training_ready_eeg(eeg_block, enabled=bool(args.training_ready))

                        fmri_block: np.ndarray | None = None
                        if fmri_full is not None:
                            if args.fmri_mode == "roi":
                                fmri_block = slice_fmri_block(fmri_full, tr=args.tr, start_sec=placement.fmri_start_sec, duration_sec=float(block_window_sec))
                                fmri_block = resample_fmri_if_needed(
                                    fmri_block,
                                    args.fmri_target_rois,
                                    args.fmri_target_t,
                                    allow_roi_resample=args.allow_fmri_roi_resample,
                                    allow_time_resample=args.allow_fmri_time_resample,
                                )
                            else:
                                fmri_block = slice_fmri_volume_block(fmri_full, tr=args.tr, start_sec=placement.fmri_start_sec, duration_sec=float(block_window_sec))
                            fmri_block = prepare_training_ready_fmri(fmri_block, fmri_mode=args.fmri_mode, enabled=bool(args.training_ready))

                        if args.label_mode == "binary_rest_task":
                            label, label_name = resolve_binary_label(trial_type)
                        else:
                            label = TASK_LABELS[task]
                            label_name = task

                        sample_id = f"{args.dataset_name}_{canonical_subject}_{task_name}_block-{block_idx:02d}_win-{sub_idx:02d}"
                        if args.pack_subject_files:
                            packed_eeg_samples.append(eeg_block.astype(np.float32))
                            if fmri_block is not None:
                                packed_fmri_samples.append(fmri_block.astype(np.float32))
                            packed_labels.append(int(label))
                            packed_sample_ids.append(sample_id)
                            packed_tasks.append(task_name)
                            packed_trial_types.append(trial_type)
                        else:
                            eeg_out_path = eeg_out_dir / f"{sample_id}.npy"
                            np.save(eeg_out_path, eeg_block)
                            fmri_out_path: Path | None = None
                            if fmri_block is not None:
                                fmri_out_path = fmri_out_dir / f"{sample_id}.npy"
                                np.save(fmri_out_path, fmri_block)

                            records.append(
                                build_sample_record(
                                    sample_id=sample_id,
                                    dataset=args.dataset_name,
                                    subject=canonical_subject,
                                    subject_uid=subject_uid,
                                    original_subject=subject,
                                    task=task_name,
                                    trial_type=trial_type,
                                    eeg_rel_path=eeg_out_path.relative_to(out_root),
                                    fmri_rel_path=None if fmri_out_path is None else fmri_out_path.relative_to(out_root),
                                    label=label,
                                    label_name=label_name,
                                    eeg=eeg_block,
                                    fmri=fmri_block,
                                    training_ready=bool(args.training_ready),
                                )
                            )

        if args.pack_subject_files and packed_eeg_samples:
            packed_eeg = stack_subject_samples(packed_eeg_samples, name="EEG")
            packed_labels_array = np.asarray(packed_labels, dtype=np.int64)
            packed_sample_ids_array = np.asarray(packed_sample_ids)
            packed_tasks_array = np.asarray(packed_tasks)
            packed_trial_types_array = np.asarray(packed_trial_types)

            arrays_to_write: dict[str, np.ndarray] = {
                "eeg": packed_eeg,
                "labels": packed_labels_array,
                "sample_id": packed_sample_ids_array,
                "task": packed_tasks_array,
                "trial_type": packed_trial_types_array,
            }
            if packed_fmri_samples:
                packed_fmri = stack_subject_samples(packed_fmri_samples, name="fMRI")
                arrays_to_write["fmri"] = packed_fmri

            subject_path = write_subject_memmap_pack(
                packed_out_dir / subject_uid,
                arrays_to_write,
            )
            subject_records.append(
                SubjectRecord(
                    dataset=args.dataset_name,
                    subject=canonical_subject,
                    subject_uid=subject_uid,
                    original_subject=subject,
                    subject_path=subject_path.relative_to(out_root).as_posix(),
                    sample_count=int(packed_labels_array.shape[0]),
                    eeg_shape="x".join(str(dim) for dim in packed_eeg.shape),
                    fmri_shape="" if not packed_fmri_samples else "x".join(str(dim) for dim in packed_fmri.shape),
                    label_shape="x".join(str(dim) for dim in packed_labels_array.shape),
                    training_ready=bool(args.training_ready),
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
    write_subject_mapping(subject_mapping_rows, out_root / "subject_mapping.csv")
    write_channel_metadata(dataset_channel_rows, out_root / "eeg_channels_dataset.csv")
    write_channel_metadata(
        [
            {"target_channel_index": index, "target_channel_name": channel_name}
            for index, channel_name in enumerate(target_channel_names)
        ],
        out_root / "eeg_channels_target.csv",
    )
    write_channel_metadata(channel_mapping_rows, out_root / "eeg_channel_mapping.csv")

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
