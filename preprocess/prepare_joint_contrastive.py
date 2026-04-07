from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from prepare_ds00233x import discover_task_recordings, load_eeg as load_ds002336_eeg
from prepare_ds002739 import build_eeg_trial_table, compute_common_electrodes, get_run_ids, load_eeg_data, load_eeg_events, load_electrode_template, load_fmri_events
from prepare_ds002739 import preprocess_eeg as preprocess_joint_eeg
from preprocess_common import (
    add_atlas_args,
    build_canonical_subject_map,
    add_common_fmri_args,
    add_eeg_patch_args,
    add_training_ready_arg,
    extract_roi_timeseries,
    find_subjects,
    get_atlas_labels_img,
    load_bold_volume,
    load_target_channel_names,
    make_channel_metadata_rows,
    make_subject_uid,
    prepare_training_ready_eeg,
    prepare_training_ready_fmri,
    preprocess_fmri_volume,
    reorder_eeg_channels,
    stack_subject_samples,
    write_channel_metadata,
    write_subject_mapping,
    write_subject_memmap_pack,
)

DS002739_JOINT_EXCLUDED_RUNS = {
    ("sub-05", "run-02"),
    ("sub-14", "run-01"),
    ("sub-14", "run-02"),
    ("sub-15", "run-02"),
}


@dataclass(frozen=True)
class ContrastiveSampleRecord:
    sample_id: str
    dataset: str
    subject: str
    subject_uid: str
    original_subject: str
    task: str
    run: str
    eeg_path: str
    fmri_path: str
    eeg_shape: str
    fmri_shape: str
    anchor_tr: int
    anchor_sec: float
    eeg_window_start_sec: float
    eeg_window_end_sec: float
    training_ready: bool = False


@dataclass(frozen=True)
class ContrastiveSubjectRecord:
    dataset: str
    subject: str
    subject_uid: str
    original_subject: str
    subject_path: str
    sample_count: int
    eeg_shape: str
    fmri_shape: str
    training_ready: bool = False


@dataclass(frozen=True)
class ContrastiveRunSummary:
    dataset: str
    subject: str
    subject_uid: str
    original_subject: str
    task: str
    run: str
    eeg_shape: str
    fmri_shape: str
    eeg_sfreq_hz: float
    fmri_target_tr_sec: float
    eeg_fmri_offset_sec: float
    candidate_trs: int
    exported_pairs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare multi-dataset EEG-fMRI contrastive pretraining pairs using TR-anchored continuous alignment.")
    parser.add_argument("--datasets", nargs="+", choices=["ds002336", "ds002338", "ds002739"], default=["ds002336", "ds002739"], help="Datasets to include in joint contrastive pretraining.")
    parser.add_argument("--ds002336-root", type=Path, default=None, help="Path to ds002336 root.")
    parser.add_argument("--ds002338-root", type=Path, default=None, help="Path to ds002338 root.")
    parser.add_argument("--ds002739-root", type=Path, default=None, help="Path to ds002739 root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output directory for unified contrastive cache.")
    parser.add_argument("--ds002336-subjects", nargs="+", default=None, help="Optional subject IDs for ds002336.")
    parser.add_argument("--ds002338-subjects", nargs="+", default=None, help="Optional subject IDs for ds002338.")
    parser.add_argument("--ds002739-subjects", nargs="+", default=None, help="Optional subject IDs for ds002739.")
    parser.add_argument("--ds002336-tasks", nargs="+", default=["motorloc", "MIpre", "MIpost", "eegNF", "fmriNF", "eegfmriNF"], help="Optional ds002336 tasks to include.")
    parser.add_argument("--ds002338-tasks", nargs="+", default=["MIpre", "MIpost", "1dNF", "2dNF"], help="Optional ds002338 tasks to include.")
    parser.add_argument("--ds002739-runs", nargs="+", default=None, help="Optional ds002739 runs to include, such as run-01 run-02.")
    parser.add_argument("--ds002739-fmri-event-type", default="dot_stim_validtrials", help="fMRI event type used to estimate ds002739 EEG/fMRI offset.")
    parser.add_argument("--eeg-window-sec", type=float, default=8.0, help="Continuous EEG context length in seconds taken before each fMRI TR anchor.")
    add_atlas_args(parser)
    add_common_fmri_args(
        parser,
        default_fmri_mode="volume",
        tr_help="Target fMRI repetition time in seconds after preprocessing.",
        standardize_help="Apply standardization inside the ROI masker when fmri-mode=roi.",
        fmri_max_shape_help="Maximum center-cropped spatial shape after resampling. Dimensions smaller than this are not padded during preprocessing.",
    )
    add_eeg_patch_args(
        parser,
        default_eeg_mode="patched",
        default_seq_len=None,
        default_patch_len=None,
        seq_len_help="EEG patch count. Defaults to round(eeg_window_sec) when omitted.",
        patch_len_help="EEG patch length. Defaults to round(eeg_target_sfreq) when omitted.",
    )
    parser.add_argument("--eeg-target-sfreq", type=float, default=200.0, help="Target EEG sampling rate in Hz.")
    parser.add_argument("--eeg-lfreq", type=float, default=0.5, help="EEG band-pass low cutoff in Hz.")
    parser.add_argument("--eeg-hfreq", type=float, default=40.0, help="EEG band-pass high cutoff in Hz.")
    parser.add_argument("--ds002336-drop-ecg", action="store_true", help="Drop ECG and other non-EEG channels when reading ds002336 BrainVision files.")
    parser.add_argument("--ds002338-drop-ecg", action="store_true", help="Drop ECG and other non-EEG channels when reading ds002338 BrainVision files.")
    parser.add_argument("--ds002336-fmri-source", default="spm_smoothed", choices=["raw", "spm_unsmoothed", "spm_smoothed"], help="fMRI source for ds002336.")
    parser.add_argument("--ds002338-fmri-source", default="spm_smoothed", choices=["raw", "spm_unsmoothed", "spm_smoothed"], help="fMRI source for ds002338.")
    parser.add_argument("--ds002336-fmri-preproc-root", type=Path, default=None, help="Optional root for ds002336 SPM preprocessed fMRI.")
    parser.add_argument("--ds002338-fmri-preproc-root", type=Path, default=None, help="Optional root for ds002338 SPM preprocessed fMRI.")
    parser.add_argument("--ds002336-discard-initial-trs", type=int, default=1, help="Initial ds002336 BOLD volumes to discard before pairing.")
    parser.add_argument("--ds002338-discard-initial-trs", type=int, default=1, help="Initial ds002338 BOLD volumes to discard before pairing.")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel worker count for subject-level preprocessing across ds002336/ds002338/ds002739. Applies to both full rebuild and incremental append runs.")
    parser.add_argument("--skip-existing-datasets", action="store_true", help="When cache already has selected datasets, skip regenerating those datasets unless force-refresh-datasets is set.")
    parser.add_argument("--force-refresh-datasets", nargs="+", choices=["ds002336", "ds002338", "ds002739"], default=[], help="Datasets to force refresh even when cache data already exists.")
    parser.add_argument(
        "--pack-subject-files",
        action="store_true",
        help="Pack all exported samples of the same normalized subject_uid into one directory of memmap-friendly NPY files.",
    )
    add_training_ready_arg(parser)
    return parser.parse_args()


def resolve_joint_patch_params(args: argparse.Namespace) -> tuple[int, int]:
    patch_len = int(args.eeg_patch_len) if args.eeg_patch_len is not None else int(round(float(args.eeg_target_sfreq)))
    seq_len = int(args.eeg_seq_len) if args.eeg_seq_len is not None else int(round(float(args.eeg_window_sec)))
    if patch_len <= 0 or seq_len <= 0:
        raise ValueError(f"Invalid EEG patch params: seq_len={seq_len}, patch_len={patch_len}")
    return seq_len, patch_len


def maybe_patch_eeg(data: np.ndarray, seq_len: int, patch_len: int) -> np.ndarray:
    target_len = seq_len * patch_len
    current_len = int(data.shape[1])
    if current_len > target_len:
        data = data[:, -target_len:]
    elif current_len < target_len:
        padded = np.zeros((data.shape[0], target_len), dtype=np.float32)
        padded[:, -current_len:] = data.astype(np.float32)
        data = padded
    return data.reshape(data.shape[0], seq_len, patch_len).astype(np.float32)


def slice_eeg_window(data: np.ndarray, sfreq: float, end_sec: float, duration_sec: float) -> np.ndarray:
    end = int(round(end_sec * sfreq))
    length = int(round(duration_sec * sfreq))
    start = end - length
    if start < 0 or end > data.shape[1]:
        raise ValueError(f"EEG slice out of range: start={start}, end={end}, total={data.shape[1]}")
    return data[:, start:end].astype(np.float32)


def slice_single_tr_fmri(series: np.ndarray, tr_index: int, fmri_mode: str) -> np.ndarray:
    if str(fmri_mode).strip().lower() == "roi":
        if tr_index < 0 or tr_index >= series.shape[1]:
            raise ValueError(f"fMRI ROI TR index out of range: {tr_index} for {series.shape}")
        return series[:, tr_index : tr_index + 1].astype(np.float32)
    if tr_index < 0 or tr_index >= series.shape[3]:
        raise ValueError(f"fMRI volume TR index out of range: {tr_index} for {series.shape}")
    return series[:, :, :, tr_index : tr_index + 1].astype(np.float32)


def event_onset_to_tr_index(onset_sec: float, tr: float, total_trs: int) -> int:
    tr_index = int(np.floor(float(onset_sec) / float(tr) + 1e-8))
    if tr_index < 0 or tr_index >= int(total_trs):
        raise ValueError(f"Event onset {onset_sec} sec maps to invalid TR index {tr_index} for total_trs={total_trs}")
    return tr_index


def build_joint_sample_record(
    *,
    sample_id: str,
    dataset: str,
    canonical_subject: str,
    subject_uid: str,
    original_subject: str,
    task: str,
    run: str,
    eeg_rel_path: Path,
    fmri_rel_path: Path,
    eeg: np.ndarray,
    fmri: np.ndarray,
    anchor_tr: int,
    anchor_sec: float,
    eeg_window_start_sec: float,
    eeg_window_end_sec: float,
    training_ready: bool,
) -> ContrastiveSampleRecord:
    return ContrastiveSampleRecord(
        sample_id=sample_id,
        dataset=dataset,
        subject=canonical_subject,
        subject_uid=subject_uid,
        original_subject=original_subject,
        task=task,
        run=run,
        eeg_path=eeg_rel_path.as_posix(),
        fmri_path=fmri_rel_path.as_posix(),
        eeg_shape="x".join(str(dim) for dim in eeg.shape),
        fmri_shape="x".join(str(dim) for dim in fmri.shape),
        anchor_tr=int(anchor_tr),
        anchor_sec=float(anchor_sec),
        eeg_window_start_sec=float(eeg_window_start_sec),
        eeg_window_end_sec=float(eeg_window_end_sec),
        training_ready=bool(training_ready),
    )


def build_ds002336_channel_order(ds_root: Path, subjects: list[str], tasks: list[str], drop_ecg: bool) -> list[str]:
    from prepare_ds00233x import compute_common_eeg_channels

    return compute_common_eeg_channels(ds_root, subjects, tasks, drop_ecg=drop_ecg)


def resolve_ds00233x_joint_fmri_path(ds_root: Path, subject: str, task: str, args: argparse.Namespace, dataset_prefix: str) -> Path:
    fmri_source = str(getattr(args, f"{dataset_prefix}_fmri_source")).strip().lower()
    if fmri_source == "raw":
        return ds_root / subject / "func" / f"{subject}_task-{task}_bold.nii.gz"

    fmri_preproc_root = getattr(args, f"{dataset_prefix}_fmri_preproc_root") or (ds_root / "derivatives" / "spm12_preproc")
    subject_dir = fmri_preproc_root / subject
    if fmri_source == "spm_smoothed":
        flat_final = subject_dir / f"{subject}_task-{task}_bold.nii"
        if flat_final.exists():
            return flat_final
        legacy_task_dir = subject_dir / f"task-{task}"
        legacy_final = legacy_task_dir / "fmri_final.nii"
        if legacy_final.exists():
            return legacy_final
        return legacy_task_dir / f"swratrim_{subject}_task-{task}_bold.nii"

    legacy_task_dir = subject_dir / f"task-{task}"
    return legacy_task_dir / f"wratrim_{subject}_task-{task}_bold.nii"


def intersect_channel_orders(dataset_channel_orders: dict[str, list[str]], dataset_order: list[str]) -> list[str]:
    common: set[str] | None = None
    for dataset_name in dataset_order:
        channels = dataset_channel_orders[dataset_name]
        common = set(channels) if common is None else (common & set(channels))
    if common is None or not common:
        raise RuntimeError("No shared EEG channels remain across the selected datasets.")
    anchor_order = dataset_channel_orders[dataset_order[0]]
    return [channel for channel in anchor_order if channel in common]


def export_sample(
    *,
    out_root: Path,
    pack_subject_files: bool,
    subject_uid: str,
    sample_id: str,
    eeg_window: np.ndarray,
    fmri_window: np.ndarray,
    packed_eeg_samples: list[np.ndarray],
    packed_fmri_samples: list[np.ndarray],
    packed_sample_ids: list[str],
    packed_dataset_names: list[str],
    packed_tasks: list[str],
    packed_runs: list[str],
    record: ContrastiveSampleRecord,
    records: list[ContrastiveSampleRecord],
) -> None:
    if pack_subject_files:
        packed_eeg_samples.append(eeg_window.astype(np.float32))
        packed_fmri_samples.append(fmri_window.astype(np.float32))
        packed_sample_ids.append(sample_id)
        packed_dataset_names.append(record.dataset)
        packed_tasks.append(record.task)
        packed_runs.append(record.run)
        return
    eeg_out_path = out_root / "eeg" / f"{sample_id}.npy"
    fmri_out_path = out_root / "fmri" / f"{sample_id}.npy"
    np.save(eeg_out_path, eeg_window.astype(np.float32))
    np.save(fmri_out_path, fmri_window.astype(np.float32))
    records.append(
        build_joint_sample_record(
            sample_id=sample_id,
            dataset=record.dataset,
            canonical_subject=record.subject,
            subject_uid=subject_uid,
            original_subject=record.original_subject,
            task=record.task,
            run=record.run,
            eeg_rel_path=eeg_out_path.relative_to(out_root),
            fmri_rel_path=fmri_out_path.relative_to(out_root),
            eeg=eeg_window,
            fmri=fmri_window,
            anchor_tr=record.anchor_tr,
            anchor_sec=record.anchor_sec,
            eeg_window_start_sec=record.eeg_window_start_sec,
            eeg_window_end_sec=record.eeg_window_end_sec,
            training_ready=record.training_ready,
        )
    )


def prepare_joint_subject_pack(
    out_root: Path,
    subject_uid: str,
    packed_eeg_samples: list[np.ndarray],
    packed_fmri_samples: list[np.ndarray],
    packed_sample_ids: list[str],
    packed_dataset_names: list[str],
    packed_tasks: list[str],
    packed_runs: list[str],
    dataset: str,
    canonical_subject: str,
    original_subject: str,
    training_ready: bool,
) -> ContrastiveSubjectRecord | None:
    if not packed_eeg_samples:
        return None
    packed_eeg = stack_subject_samples(packed_eeg_samples, name="EEG")
    packed_fmri = stack_subject_samples(packed_fmri_samples, name="fMRI")
    subject_path = write_subject_memmap_pack(
        out_root / "subjects" / subject_uid,
        {
            "eeg": packed_eeg,
            "fmri": packed_fmri,
            "sample_id": np.asarray(packed_sample_ids),
            "dataset": np.asarray(packed_dataset_names),
            "task": np.asarray(packed_tasks),
            "run": np.asarray(packed_runs),
        },
    )
    return ContrastiveSubjectRecord(
        dataset=dataset,
        subject=canonical_subject,
        subject_uid=subject_uid,
        original_subject=original_subject,
        subject_path=subject_path.relative_to(out_root).as_posix(),
        sample_count=int(len(packed_sample_ids)),
        eeg_shape="x".join(str(dim) for dim in packed_eeg.shape),
        fmri_shape="x".join(str(dim) for dim in packed_fmri.shape),
        training_ready=bool(training_ready),
    )


def remap_subject_pack_eeg_channels(out_root: Path, subject_rel_path: str, keep_channel_indices: list[int]) -> str:
    subject_dir = out_root / subject_rel_path
    eeg_path = subject_dir / "eeg.npy"
    eeg = np.load(eeg_path, mmap_mode="r", allow_pickle=False)
    if eeg.ndim < 2:
        raise ValueError(f"Subject pack EEG must have at least 2 dims [N,C,...], got {eeg.shape} at {subject_dir}")
    if eeg.shape[1] == len(keep_channel_indices) and keep_channel_indices == list(range(eeg.shape[1])):
        return "x".join(str(dim) for dim in eeg.shape)

    arrays: dict[str, np.ndarray] = {}
    for array_path in sorted(subject_dir.glob("*.npy")):
        arrays[array_path.stem] = np.asarray(np.load(array_path, allow_pickle=False))
    arrays["eeg"] = np.asarray(eeg[:, keep_channel_indices, ...], dtype=np.float32)
    updated_path = write_subject_memmap_pack(subject_dir, arrays)
    updated_eeg = np.load(updated_path / "eeg.npy", mmap_mode="r", allow_pickle=False)
    return "x".join(str(dim) for dim in updated_eeg.shape)


def prepare_joint_dataset_ds33x(
    *,
    dataset_name: str,
    ds_root: Path,
    subjects: list[str],
    canonical_map: dict[str, str],
    target_channel_names: list[str],
    out_root: Path,
    labels_img: str,
    seq_len: int,
    patch_len: int,
    args: argparse.Namespace,
) -> tuple[list[ContrastiveSampleRecord], list[ContrastiveSubjectRecord], list[ContrastiveRunSummary], list[dict[str, object]]]:
    def process_subject_ds33x(original_subject: str) -> tuple[list[ContrastiveSampleRecord], ContrastiveSubjectRecord | None, list[ContrastiveRunSummary], list[dict[str, object]]]:
        subject_records_local: list[ContrastiveSampleRecord] = []
        subject_summaries_local: list[ContrastiveRunSummary] = []
        subject_channel_mapping_rows: list[dict[str, object]] = []

        canonical_subject = canonical_map[original_subject]
        subject_uid = make_subject_uid(dataset_name, canonical_subject)
        packed_eeg_samples: list[np.ndarray] = []
        packed_fmri_samples: list[np.ndarray] = []
        packed_sample_ids: list[str] = []
        packed_dataset_names: list[str] = []
        packed_tasks: list[str] = []
        packed_runs: list[str] = []

        recording_args = argparse.Namespace(
            fmri_source=getattr(args, f"{dataset_name}_fmri_source"),
            fmri_preproc_root=getattr(args, f"{dataset_name}_fmri_preproc_root"),
            eeg_only=False,
        )

        for task in getattr(args, f"{dataset_name}_tasks"):
            recordings = discover_task_recordings(ds_root, original_subject, task, recording_args)
            for recording in recordings:
                if not recording.eeg_vhdr.exists() or not recording.fmri_nii.exists():
                    continue
                raw_eeg, raw_sfreq, eeg_protocol_start_sec, eeg_channel_names = load_ds002336_eeg(recording.eeg_vhdr, drop_ecg=bool(getattr(args, f"{dataset_name}_drop_ecg")))
                reordered_preview, mapping_rows = reorder_eeg_channels(raw_eeg, eeg_channel_names, target_channel_names)
                if (not subject_channel_mapping_rows) and mapping_rows:
                    subject_channel_mapping_rows.extend({"dataset": dataset_name, **row} for row in mapping_rows)
                eeg_data, processed_sfreq = preprocess_joint_eeg(reordered_preview, source_sfreq=raw_sfreq, args=args)

                fmri_discard_initial_trs = int(getattr(args, f"{dataset_name}_discard_initial_trs"))
                if str(getattr(args, f"{dataset_name}_fmri_source")) != "raw":
                    fmri_discard_initial_trs = 0
                if args.fmri_mode == "roi":
                    fmri_source, _, _ = extract_roi_timeseries(
                        fmri_nii_path=recording.fmri_nii,
                        labels_img=labels_img,
                        tr=args.tr,
                        standardize_fmri=args.standardize_fmri,
                        discard_initial_trs=fmri_discard_initial_trs,
                        include_metadata=True,
                    )
                    total_trs = int(fmri_source.shape[1])
                else:
                    raw_fmri, _, voxel_size = load_bold_volume(recording.fmri_nii, discard_initial_trs=fmri_discard_initial_trs, include_metadata=True)
                    fmri_source = preprocess_fmri_volume(
                        raw_fmri,
                        voxel_size=voxel_size,
                        source_tr=float(voxel_size[3]),
                        target_voxel_size=args.fmri_voxel_size,
                        target_tr=float(args.tr),
                        max_shape=args.fmri_max_shape,
                        use_float16=bool(args.fmri_float16),
                    )
                    total_trs = int(fmri_source.shape[3])

                exported_pairs = 0
                for tr_index in range(total_trs):
                    anchor_sec = float(tr_index) * float(args.tr)
                    eeg_window_end_sec = float(eeg_protocol_start_sec) + anchor_sec
                    eeg_window_start_sec = eeg_window_end_sec - float(args.eeg_window_sec)
                    try:
                        eeg_window = slice_eeg_window(eeg_data, sfreq=processed_sfreq, end_sec=eeg_window_end_sec, duration_sec=float(args.eeg_window_sec))
                        fmri_window = slice_single_tr_fmri(fmri_source, tr_index=tr_index, fmri_mode=args.fmri_mode)
                    except ValueError:
                        continue
                    if args.eeg_mode == "patched":
                        eeg_window = maybe_patch_eeg(eeg_window, seq_len=seq_len, patch_len=patch_len)
                    eeg_window = prepare_training_ready_eeg(eeg_window, enabled=bool(args.training_ready))
                    fmri_window = prepare_training_ready_fmri(fmri_window, fmri_mode=args.fmri_mode, enabled=bool(args.training_ready))
                    sample_id = f"{dataset_name}_{canonical_subject}_{recording.task_name}_tr-{tr_index:04d}"
                    record_template = ContrastiveSampleRecord(
                        sample_id=sample_id,
                        dataset=dataset_name,
                        subject=canonical_subject,
                        subject_uid=subject_uid,
                        original_subject=original_subject,
                        task=recording.task_key,
                        run=recording.run or recording.task_name,
                        eeg_path="",
                        fmri_path="",
                        eeg_shape="x".join(str(dim) for dim in eeg_window.shape),
                        fmri_shape="x".join(str(dim) for dim in fmri_window.shape),
                        anchor_tr=tr_index,
                        anchor_sec=anchor_sec,
                        eeg_window_start_sec=eeg_window_start_sec,
                        eeg_window_end_sec=eeg_window_end_sec,
                        training_ready=bool(args.training_ready),
                    )
                    export_sample(
                        out_root=out_root,
                        pack_subject_files=bool(args.pack_subject_files),
                        subject_uid=subject_uid,
                        sample_id=sample_id,
                        eeg_window=eeg_window,
                        fmri_window=fmri_window,
                        packed_eeg_samples=packed_eeg_samples,
                        packed_fmri_samples=packed_fmri_samples,
                        packed_sample_ids=packed_sample_ids,
                        packed_dataset_names=packed_dataset_names,
                        packed_tasks=packed_tasks,
                        packed_runs=packed_runs,
                        record=record_template,
                        records=subject_records_local,
                    )
                    exported_pairs += 1

                subject_summaries_local.append(
                    ContrastiveRunSummary(
                        dataset=dataset_name,
                        subject=canonical_subject,
                        subject_uid=subject_uid,
                        original_subject=original_subject,
                        task=recording.task_key,
                        run=recording.run or recording.task_name,
                        eeg_shape="x".join(str(dim) for dim in eeg_data.shape),
                        fmri_shape="x".join(str(dim) for dim in fmri_source.shape),
                        eeg_sfreq_hz=processed_sfreq,
                        fmri_target_tr_sec=float(args.tr),
                        eeg_fmri_offset_sec=float(eeg_protocol_start_sec),
                        candidate_trs=total_trs,
                        exported_pairs=exported_pairs,
                    )
                )

        subject_record_local = prepare_joint_subject_pack(
            out_root=out_root,
            subject_uid=subject_uid,
            packed_eeg_samples=packed_eeg_samples,
            packed_fmri_samples=packed_fmri_samples,
            packed_sample_ids=packed_sample_ids,
            packed_dataset_names=packed_dataset_names,
            packed_tasks=packed_tasks,
            packed_runs=packed_runs,
            dataset=dataset_name,
            canonical_subject=canonical_subject,
            original_subject=original_subject,
            training_ready=bool(args.training_ready),
        )

        return subject_records_local, subject_record_local, subject_summaries_local, subject_channel_mapping_rows

    local_records: list[ContrastiveSampleRecord] = []
    local_subject_records: list[ContrastiveSubjectRecord] = []
    local_summaries: list[ContrastiveRunSummary] = []
    local_channel_mapping_rows: list[dict[str, object]] = []

    worker_count = min(max(1, int(args.num_workers)), max(1, len(subjects)))
    if worker_count > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(process_subject_ds33x, original_subject) for original_subject in subjects]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Preparing {dataset_name} joint contrastive"):
                subject_records_local, subject_record_local, subject_summaries_local, subject_channel_mapping_rows = future.result()
                local_records.extend(subject_records_local)
                local_summaries.extend(subject_summaries_local)
                if subject_record_local is not None:
                    local_subject_records.append(subject_record_local)
                if (not local_channel_mapping_rows) and subject_channel_mapping_rows:
                    local_channel_mapping_rows.extend(subject_channel_mapping_rows)
    else:
        for original_subject in tqdm(subjects, desc=f"Preparing {dataset_name} joint contrastive"):
            subject_records_local, subject_record_local, subject_summaries_local, subject_channel_mapping_rows = process_subject_ds33x(original_subject)
            local_records.extend(subject_records_local)
            local_summaries.extend(subject_summaries_local)
            if subject_record_local is not None:
                local_subject_records.append(subject_record_local)
            if (not local_channel_mapping_rows) and subject_channel_mapping_rows:
                local_channel_mapping_rows.extend(subject_channel_mapping_rows)

    return local_records, local_subject_records, local_summaries, local_channel_mapping_rows


def main() -> None:
    args = parse_args()
    if args.eeg_window_sec <= 0:
        raise ValueError("--eeg-window-sec must be positive")

    out_root = args.output_root
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

    labels_img = get_atlas_labels_img(args.atlas_labels_img, atlas_cache_dir, args.n_rois) if args.fmri_mode == "roi" else ""
    seq_len, patch_len = resolve_joint_patch_params(args)

    dataset_order = list(dict.fromkeys(args.datasets))
    refresh_datasets = set(args.force_refresh_datasets or [])
    existing_manifest_path = out_root / "manifest_all.csv"
    existing_run_summary_path = out_root / "run_summary.csv"
    existing_target_channel_path = out_root / "eeg_channels_target.csv"
    existing_subject_mapping_path = out_root / "subject_mapping.csv"
    existing_dataset_channel_path = out_root / "eeg_channels_dataset.csv"
    existing_channel_mapping_path = out_root / "eeg_channel_mapping.csv"

    preserved_subject_rows: list[dict[str, object]] = []
    preserved_summary_rows: list[dict[str, object]] = []
    preserved_subject_mapping_rows: list[dict[str, object]] = []
    preserved_dataset_channel_rows: list[dict[str, object]] = []
    preserved_channel_mapping_rows: list[dict[str, object]] = []
    existing_cached_datasets: set[str] = set()
    if args.skip_existing_datasets and existing_manifest_path.exists():
        existing_manifest = pd.read_csv(existing_manifest_path)
        if "dataset" in existing_manifest.columns:
            existing_cached_datasets = {str(value) for value in existing_manifest["dataset"].dropna().unique().tolist()}
            preserved_subject_rows = existing_manifest.to_dict("records")
        if existing_run_summary_path.exists():
            preserved_summary_rows = pd.read_csv(existing_run_summary_path).to_dict("records")
        if existing_subject_mapping_path.exists():
            preserved_subject_mapping_rows = pd.read_csv(existing_subject_mapping_path).to_dict("records")
        if existing_dataset_channel_path.exists():
            preserved_dataset_channel_rows = pd.read_csv(existing_dataset_channel_path).to_dict("records")
        if existing_channel_mapping_path.exists():
            preserved_channel_mapping_rows = pd.read_csv(existing_channel_mapping_path).to_dict("records")

    dataset_order_to_process = [
        dataset_name
        for dataset_name in dataset_order
        if (dataset_name not in existing_cached_datasets) or (dataset_name in refresh_datasets) or (not args.skip_existing_datasets)
    ]
    skipped_existing_datasets = set(dataset_order) - set(dataset_order_to_process)

    if preserved_subject_rows:
        preserved_subject_rows = [row for row in preserved_subject_rows if str(row.get("dataset", "")) not in set(dataset_order_to_process)]
    if preserved_summary_rows:
        preserved_summary_rows = [row for row in preserved_summary_rows if str(row.get("dataset", "")) not in set(dataset_order_to_process)]
    if preserved_subject_mapping_rows:
        preserved_subject_mapping_rows = [row for row in preserved_subject_mapping_rows if str(row.get("dataset", "")) not in set(dataset_order_to_process)]
    if preserved_dataset_channel_rows:
        preserved_dataset_channel_rows = [row for row in preserved_dataset_channel_rows if str(row.get("dataset", "")) not in set(dataset_order_to_process)]
    if preserved_channel_mapping_rows:
        preserved_channel_mapping_rows = [row for row in preserved_channel_mapping_rows if str(row.get("dataset", "")) not in set(dataset_order_to_process)]

    dataset_channel_orders: dict[str, list[str]] = {}
    dataset_subjects: dict[str, list[str]] = {}
    dataset_canonical_subject_maps: dict[str, dict[str, str]] = {}
    subject_mapping_rows: list[dict[str, object]] = []
    dataset_channel_rows: list[dict[str, object]] = []

    for dataset_name in ["ds002336", "ds002338"]:
        if dataset_name in dataset_order_to_process:
            ds_root = getattr(args, f"{dataset_name}_root")
            if ds_root is None:
                raise ValueError(f"--{dataset_name}-root is required when datasets include {dataset_name}")
            subjects = find_subjects(ds_root, getattr(args, f"{dataset_name}_subjects"))
            dataset_subjects[dataset_name] = subjects
            dataset_canonical_subject_maps[dataset_name] = build_canonical_subject_map(subjects)
            dataset_channel_orders[dataset_name] = build_ds002336_channel_order(
                ds_root,
                subjects,
                list(getattr(args, f"{dataset_name}_tasks")),
                drop_ecg=bool(getattr(args, f"{dataset_name}_drop_ecg")),
            )
            dataset_channel_rows.extend(
                {"dataset": dataset_name, "channel_name": channel_name, "target_channel_name": channel_name}
                for channel_name in dataset_channel_orders[dataset_name]
            )
            subject_mapping_rows.extend(
                {
                    "dataset": dataset_name,
                    "original_subject": subject,
                    "subject": dataset_canonical_subject_maps[dataset_name][subject],
                    "subject_uid": make_subject_uid(dataset_name, dataset_canonical_subject_maps[dataset_name][subject]),
                }
                for subject in subjects
            )

    if "ds002739" in dataset_order_to_process:
        if args.ds002739_root is None:
            raise ValueError("--ds002739-root is required when datasets include ds002739")
        ds002739_root = args.ds002739_root
        subjects = find_subjects(ds002739_root, args.ds002739_subjects)
        dataset_subjects["ds002739"] = subjects
        dataset_canonical_subject_maps["ds002739"] = build_canonical_subject_map(subjects)
        electrode_template = load_electrode_template(ds002739_root)
        dataset_channel_orders["ds002739"] = compute_common_electrodes(ds002739_root, subjects, args.ds002739_runs, electrode_template)
        dataset_channel_rows.extend(make_channel_metadata_rows("ds002739", electrode_template))
        subject_mapping_rows.extend(
            {
                "dataset": "ds002739",
                "original_subject": subject,
                "subject": dataset_canonical_subject_maps["ds002739"][subject],
                "subject_uid": make_subject_uid("ds002739", dataset_canonical_subject_maps["ds002739"][subject]),
            }
            for subject in subjects
        )

    previous_target_channel_names = load_target_channel_names(existing_target_channel_path) if existing_target_channel_path.exists() else None
    if previous_target_channel_names and skipped_existing_datasets:
        if dataset_channel_orders:
            common = set(previous_target_channel_names)
            for channels in dataset_channel_orders.values():
                common &= set(channels)
            target_channel_names = [channel for channel in previous_target_channel_names if channel in common]
            if not target_channel_names:
                raise RuntimeError("No shared EEG channels remain after merging existing cache channels with newly processed datasets.")
        else:
            target_channel_names = list(previous_target_channel_names)
    else:
        if not dataset_channel_orders:
            raise RuntimeError("No datasets were selected for processing and no existing channel manifest was found.")
        target_channel_names = intersect_channel_orders(dataset_channel_orders, dataset_order_to_process)

    records: list[ContrastiveSampleRecord] = []
    subject_records: list[ContrastiveSubjectRecord] = []
    summaries: list[ContrastiveRunSummary] = []
    channel_mapping_rows: list[dict[str, object]] = []

    if not dataset_order_to_process:
        print("No dataset requires regeneration. Reusing existing cached datasets.")

    ds33x_datasets = [dataset_name for dataset_name in dataset_order_to_process if dataset_name in {"ds002336", "ds002338"}]
    if ds33x_datasets:
        worker_count = min(max(1, int(args.num_workers)), len(ds33x_datasets))
        if worker_count > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        prepare_joint_dataset_ds33x,
                        dataset_name=dataset_name,
                        ds_root=getattr(args, f"{dataset_name}_root"),
                        subjects=dataset_subjects[dataset_name],
                        canonical_map=dataset_canonical_subject_maps[dataset_name],
                        target_channel_names=target_channel_names,
                        out_root=out_root,
                        labels_img=labels_img,
                        seq_len=seq_len,
                        patch_len=patch_len,
                        args=args,
                    ): dataset_name
                    for dataset_name in ds33x_datasets
                }
                for future in concurrent.futures.as_completed(futures):
                    local_records, local_subject_records, local_summaries, local_mapping_rows = future.result()
                    records.extend(local_records)
                    subject_records.extend(local_subject_records)
                    summaries.extend(local_summaries)
                    channel_mapping_rows.extend(local_mapping_rows)
        else:
            for dataset_name in ds33x_datasets:
                local_records, local_subject_records, local_summaries, local_mapping_rows = prepare_joint_dataset_ds33x(
                    dataset_name=dataset_name,
                    ds_root=getattr(args, f"{dataset_name}_root"),
                    subjects=dataset_subjects[dataset_name],
                    canonical_map=dataset_canonical_subject_maps[dataset_name],
                    target_channel_names=target_channel_names,
                    out_root=out_root,
                    labels_img=labels_img,
                    seq_len=seq_len,
                    patch_len=patch_len,
                    args=args,
                )
                records.extend(local_records)
                subject_records.extend(local_subject_records)
                summaries.extend(local_summaries)
                channel_mapping_rows.extend(local_mapping_rows)

    for dataset_name in dataset_order_to_process:

        if dataset_name == "ds002739":
            ds_root = args.ds002739_root
            electrode_template = load_electrode_template(ds_root)
            def process_subject_ds002739(original_subject: str) -> tuple[list[ContrastiveSampleRecord], ContrastiveSubjectRecord | None, list[ContrastiveRunSummary], list[dict[str, object]]]:
                subject_records_local: list[ContrastiveSampleRecord] = []
                subject_summaries_local: list[ContrastiveRunSummary] = []
                subject_channel_mapping_rows: list[dict[str, object]] = []

                canonical_subject = dataset_canonical_subject_maps["ds002739"][original_subject]
                subject_uid = make_subject_uid("ds002739", canonical_subject)
                packed_eeg_samples = []
                packed_fmri_samples = []
                packed_sample_ids = []
                packed_dataset_names = []
                packed_tasks = []
                packed_runs = []

                func_dir = ds_root / original_subject / "func"
                eeg_dir = ds_root / original_subject / "EEG"
                for run in get_run_ids(func_dir, args.ds002739_runs):
                    if (original_subject, run) in DS002739_JOINT_EXCLUDED_RUNS:
                        continue
                    bold_path = func_dir / f"{original_subject}_task-main_{run}_bold.nii.gz"
                    fmri_events_path = func_dir / f"{original_subject}_task-main_{run}_events.tsv"
                    eeg_data_path = eeg_dir / f"EEG_data_{original_subject}_{run}.mat"
                    eeg_events_path = eeg_dir / f"EEG_events_{original_subject}_{run}.mat"
                    if not all(path.exists() for path in [bold_path, fmri_events_path, eeg_data_path, eeg_events_path]):
                        continue

                    raw_eeg, raw_sfreq, kept_electrodes = load_eeg_data(eeg_data_path, electrode_template=electrode_template)
                    reordered_preview, mapping_rows = reorder_eeg_channels(raw_eeg, kept_electrodes, target_channel_names)
                    if (not subject_channel_mapping_rows) and mapping_rows:
                        subject_channel_mapping_rows.extend({"dataset": "ds002739", **row} for row in mapping_rows)
                    eeg_data, processed_sfreq = preprocess_joint_eeg(reordered_preview, source_sfreq=raw_sfreq, args=args)

                    eeg_events = load_eeg_events(eeg_events_path)
                    eeg_trials = build_eeg_trial_table(eeg_events)
                    fmri_events = load_fmri_events(fmri_events_path, event_type=args.ds002739_fmri_event_type)
                    if eeg_trials.empty or fmri_events.empty:
                        continue
                    pair_count = min(len(eeg_trials), len(fmri_events))
                    eeg_trials = eeg_trials.iloc[:pair_count].reset_index(drop=True)
                    fmri_events = fmri_events.iloc[:pair_count].reset_index(drop=True)
                    eeg_protocol_start_sec = float(eeg_trials.iloc[0]["eeg_onset_sec"]) - float(fmri_events.iloc[0]["onset"])

                    if args.fmri_mode == "roi":
                        fmri_source, _, _ = extract_roi_timeseries(
                            fmri_nii_path=bold_path,
                            labels_img=labels_img,
                            tr=args.tr,
                            standardize_fmri=args.standardize_fmri,
                            include_metadata=True,
                        )
                        total_trs = int(fmri_source.shape[1])
                    else:
                        raw_fmri, _, voxel_size = load_bold_volume(bold_path, include_metadata=True)
                        fmri_source = preprocess_fmri_volume(
                            raw_fmri,
                            voxel_size=voxel_size,
                            source_tr=float(voxel_size[3]),
                            target_voxel_size=args.fmri_voxel_size,
                            target_tr=float(args.tr),
                            max_shape=args.fmri_max_shape,
                            use_float16=bool(args.fmri_float16),
                        )
                        total_trs = int(fmri_source.shape[3])

                    exported_pairs = 0
                    for tr_index in range(total_trs):
                        anchor_sec = float(tr_index) * float(args.tr)
                        eeg_window_end_sec = float(eeg_protocol_start_sec) + anchor_sec
                        eeg_window_start_sec = eeg_window_end_sec - float(args.eeg_window_sec)
                        try:
                            eeg_window = slice_eeg_window(eeg_data, sfreq=processed_sfreq, end_sec=eeg_window_end_sec, duration_sec=float(args.eeg_window_sec))
                            fmri_window = slice_single_tr_fmri(fmri_source, tr_index=tr_index, fmri_mode=args.fmri_mode)
                        except ValueError:
                            continue
                        if args.eeg_mode == "patched":
                            eeg_window = maybe_patch_eeg(eeg_window, seq_len=seq_len, patch_len=patch_len)
                        eeg_window = prepare_training_ready_eeg(eeg_window, enabled=bool(args.training_ready))
                        fmri_window = prepare_training_ready_fmri(fmri_window, fmri_mode=args.fmri_mode, enabled=bool(args.training_ready))
                        sample_id = f"ds002739_{canonical_subject}_{run}_tr-{tr_index:04d}"
                        record_template = ContrastiveSampleRecord(
                            sample_id=sample_id,
                            dataset="ds002739",
                            subject=canonical_subject,
                            subject_uid=subject_uid,
                            original_subject=original_subject,
                            task="main",
                            run=run,
                            eeg_path="",
                            fmri_path="",
                            eeg_shape="x".join(str(dim) for dim in eeg_window.shape),
                            fmri_shape="x".join(str(dim) for dim in fmri_window.shape),
                            anchor_tr=tr_index,
                            anchor_sec=anchor_sec,
                            eeg_window_start_sec=eeg_window_start_sec,
                            eeg_window_end_sec=eeg_window_end_sec,
                            training_ready=bool(args.training_ready),
                        )
                        export_sample(
                            out_root=out_root,
                            pack_subject_files=bool(args.pack_subject_files),
                            subject_uid=subject_uid,
                            sample_id=sample_id,
                            eeg_window=eeg_window,
                            fmri_window=fmri_window,
                            packed_eeg_samples=packed_eeg_samples,
                            packed_fmri_samples=packed_fmri_samples,
                            packed_sample_ids=packed_sample_ids,
                            packed_dataset_names=packed_dataset_names,
                            packed_tasks=packed_tasks,
                            packed_runs=packed_runs,
                            record=record_template,
                            records=subject_records_local,
                        )
                        exported_pairs += 1

                    subject_summaries_local.append(
                        ContrastiveRunSummary(
                            dataset="ds002739",
                            subject=canonical_subject,
                            subject_uid=subject_uid,
                            original_subject=original_subject,
                            task="main",
                            run=run,
                            eeg_shape="x".join(str(dim) for dim in eeg_data.shape),
                            fmri_shape="x".join(str(dim) for dim in fmri_source.shape),
                            eeg_sfreq_hz=processed_sfreq,
                            fmri_target_tr_sec=float(args.tr),
                            eeg_fmri_offset_sec=float(eeg_protocol_start_sec),
                            candidate_trs=total_trs,
                            exported_pairs=exported_pairs,
                        )
                    )

                subject_record = prepare_joint_subject_pack(
                    out_root=out_root,
                    subject_uid=subject_uid,
                    packed_eeg_samples=packed_eeg_samples,
                    packed_fmri_samples=packed_fmri_samples,
                    packed_sample_ids=packed_sample_ids,
                    packed_dataset_names=packed_dataset_names,
                    packed_tasks=packed_tasks,
                    packed_runs=packed_runs,
                    dataset="ds002739",
                    canonical_subject=canonical_subject,
                    original_subject=original_subject,
                    training_ready=bool(args.training_ready),
                )
                return subject_records_local, subject_record, subject_summaries_local, subject_channel_mapping_rows

            subjects_739 = dataset_subjects["ds002739"]
            worker_count_739 = min(max(1, int(args.num_workers)), max(1, len(subjects_739)))
            if worker_count_739 > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count_739) as executor:
                    futures = [executor.submit(process_subject_ds002739, original_subject) for original_subject in subjects_739]
                    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Preparing ds002739 joint contrastive"):
                        subject_records_local, subject_record_local, subject_summaries_local, subject_channel_mapping_rows = future.result()
                        records.extend(subject_records_local)
                        summaries.extend(subject_summaries_local)
                        if subject_record_local is not None:
                            subject_records.append(subject_record_local)
                        if (not any(row.get("dataset") == "ds002739" and "source_channel_name" in row for row in channel_mapping_rows)) and subject_channel_mapping_rows:
                            channel_mapping_rows.extend(subject_channel_mapping_rows)
            else:
                for original_subject in tqdm(subjects_739, desc="Preparing ds002739 joint contrastive"):
                    subject_records_local, subject_record_local, subject_summaries_local, subject_channel_mapping_rows = process_subject_ds002739(original_subject)
                    records.extend(subject_records_local)
                    summaries.extend(subject_summaries_local)
                    if subject_record_local is not None:
                        subject_records.append(subject_record_local)
                    if (not any(row.get("dataset") == "ds002739" and "source_channel_name" in row for row in channel_mapping_rows)) and subject_channel_mapping_rows:
                        channel_mapping_rows.extend(subject_channel_mapping_rows)

    if not records and not subject_records and not preserved_subject_rows:
        raise RuntimeError("No joint contrastive samples were exported or preserved. Check dataset roots and alignment settings.")

    final_subject_rows = preserved_subject_rows + [record.__dict__ for record in subject_records]
    final_sample_rows = [record.__dict__ for record in records]

    if final_subject_rows and previous_target_channel_names is not None and list(previous_target_channel_names) != list(target_channel_names):
        previous_index = {name: idx for idx, name in enumerate(previous_target_channel_names)}
        keep_indices = [previous_index[name] for name in target_channel_names if name in previous_index]
        if len(keep_indices) != len(target_channel_names):
            raise RuntimeError("Cannot remap existing cache channels because some new target channels were absent in previous target channel list.")
        for row in final_subject_rows:
            subject_rel_path = str(row.get("subject_path", "")).strip()
            dataset_name = str(row.get("dataset", "")).strip()
            if not subject_rel_path or dataset_name not in skipped_existing_datasets:
                continue
            updated_eeg_shape = remap_subject_pack_eeg_channels(out_root, subject_rel_path, keep_indices)
            row["eeg_shape"] = updated_eeg_shape

    if final_subject_rows:
        pd.DataFrame(final_subject_rows).sort_values(by=["dataset", "subject_uid"], kind="stable").to_csv(out_root / "manifest_all.csv", index=False)
    else:
        pd.DataFrame(final_sample_rows).sort_values(by=["sample_id"], kind="stable").to_csv(out_root / "manifest_all.csv", index=False)

    final_summary_rows = preserved_summary_rows + [summary.__dict__ for summary in summaries]
    if final_summary_rows:
        pd.DataFrame(final_summary_rows).sort_values(by=["dataset", "subject_uid", "run"], kind="stable").to_csv(out_root / "run_summary.csv", index=False)

    final_subject_mapping_rows = preserved_subject_mapping_rows + subject_mapping_rows
    write_subject_mapping(final_subject_mapping_rows, out_root / "subject_mapping.csv")

    final_dataset_channel_rows = preserved_dataset_channel_rows + dataset_channel_rows
    write_channel_metadata(final_dataset_channel_rows, out_root / "eeg_channels_dataset.csv")
    write_channel_metadata(
        [
            {
                "target_channel_index": index,
                "target_channel_name": channel_name,
            }
            for index, channel_name in enumerate(target_channel_names)
        ],
        out_root / "eeg_channels_target.csv",
    )
    final_channel_mapping_rows = preserved_channel_mapping_rows + channel_mapping_rows
    write_channel_metadata(final_channel_mapping_rows, out_root / "eeg_channel_mapping.csv")

if __name__ == "__main__":
    main()
