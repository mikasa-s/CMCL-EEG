from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from prepare_ds002336 import load_eeg as load_ds002336_eeg
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
    parser.add_argument("--datasets", nargs="+", choices=["ds002336", "ds002739"], default=["ds002336", "ds002739"], help="Datasets to include in joint contrastive pretraining.")
    parser.add_argument("--ds002336-root", type=Path, default=None, help="Path to ds002336 root.")
    parser.add_argument("--ds002739-root", type=Path, default=None, help="Path to ds002739 root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output directory for unified contrastive cache.")
    parser.add_argument("--ds002336-subjects", nargs="+", default=None, help="Optional subject IDs for ds002336.")
    parser.add_argument("--ds002739-subjects", nargs="+", default=None, help="Optional subject IDs for ds002739.")
    parser.add_argument("--ds002336-tasks", nargs="+", default=["motorloc", "MIpre", "MIpost", "eegNF", "fmriNF", "eegfmriNF"], help="Optional ds002336 tasks to include.")
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
    parser.add_argument("--ds002336-fmri-source", default="spm_smoothed", choices=["raw", "spm_unsmoothed", "spm_smoothed"], help="fMRI source for ds002336.")
    parser.add_argument("--ds002336-fmri-preproc-root", type=Path, default=None, help="Optional root for ds002336 SPM preprocessed fMRI.")
    parser.add_argument("--ds002336-discard-initial-trs", type=int, default=1, help="Initial ds002336 BOLD volumes to discard before pairing.")
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
    from prepare_ds002336 import compute_common_eeg_channels

    return compute_common_eeg_channels(ds_root, subjects, tasks, drop_ecg=drop_ecg)


def resolve_ds002336_joint_fmri_path(ds_root: Path, subject: str, task: str, args: argparse.Namespace) -> Path:
    fmri_source = str(args.ds002336_fmri_source).strip().lower()
    if fmri_source == "raw":
        return ds_root / subject / "func" / f"{subject}_task-{task}_bold.nii.gz"

    fmri_preproc_root = args.ds002336_fmri_preproc_root or (ds_root / "derivatives" / "spm12_preproc")
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
    dataset_channel_orders: dict[str, list[str]] = {}
    dataset_subjects: dict[str, list[str]] = {}
    dataset_canonical_subject_maps: dict[str, dict[str, str]] = {}
    subject_mapping_rows: list[dict[str, object]] = []
    dataset_channel_rows: list[dict[str, object]] = []

    if "ds002336" in dataset_order:
        if args.ds002336_root is None:
            raise ValueError("--ds002336-root is required when datasets include ds002336")
        ds002336_root = args.ds002336_root
        subjects = find_subjects(ds002336_root, args.ds002336_subjects)
        dataset_subjects["ds002336"] = subjects
        dataset_canonical_subject_maps["ds002336"] = build_canonical_subject_map(subjects)
        dataset_channel_orders["ds002336"] = build_ds002336_channel_order(ds002336_root, subjects, list(args.ds002336_tasks), drop_ecg=bool(args.ds002336_drop_ecg))
        dataset_channel_rows.extend(
            {"dataset": "ds002336", "channel_name": channel_name, "target_channel_name": channel_name}
            for channel_name in dataset_channel_orders["ds002336"]
        )
        subject_mapping_rows.extend(
            {
                "dataset": "ds002336",
                "original_subject": subject,
                "subject": dataset_canonical_subject_maps["ds002336"][subject],
                "subject_uid": make_subject_uid("ds002336", dataset_canonical_subject_maps["ds002336"][subject]),
            }
            for subject in subjects
        )

    if "ds002739" in dataset_order:
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

    target_channel_names = intersect_channel_orders(dataset_channel_orders, dataset_order)

    records: list[ContrastiveSampleRecord] = []
    subject_records: list[ContrastiveSubjectRecord] = []
    summaries: list[ContrastiveRunSummary] = []
    channel_mapping_rows: list[dict[str, object]] = []

    for dataset_name in dataset_order:
        if dataset_name == "ds002336":
            ds_root = args.ds002336_root
            for original_subject in tqdm(dataset_subjects["ds002336"], desc="Preparing ds002336 joint contrastive"):
                canonical_subject = dataset_canonical_subject_maps["ds002336"][original_subject]
                subject_uid = make_subject_uid("ds002336", canonical_subject)
                packed_eeg_samples: list[np.ndarray] = []
                packed_fmri_samples: list[np.ndarray] = []
                packed_sample_ids: list[str] = []
                packed_dataset_names: list[str] = []
                packed_tasks: list[str] = []
                packed_runs: list[str] = []

                for task in args.ds002336_tasks:
                    eeg_vhdr = ds_root / "derivatives" / original_subject / "eeg_pp" / f"{original_subject}_task-{task}_eeg_pp.vhdr"
                    fmri_nii = resolve_ds002336_joint_fmri_path(ds_root, original_subject, task, args)
                    if not eeg_vhdr.exists() or not fmri_nii.exists():
                        continue
                    raw_eeg, raw_sfreq, eeg_protocol_start_sec, eeg_channel_names = load_ds002336_eeg(eeg_vhdr, drop_ecg=bool(args.ds002336_drop_ecg))
                    if not any(row.get("dataset") == "ds002336" and "source_channel_name" in row for row in channel_mapping_rows):
                        reordered_preview, mapping_rows = reorder_eeg_channels(raw_eeg, eeg_channel_names, target_channel_names)
                        channel_mapping_rows.extend({"dataset": "ds002336", **row} for row in mapping_rows)
                    else:
                        reordered_preview, _ = reorder_eeg_channels(raw_eeg, eeg_channel_names, target_channel_names)
                    eeg_data, processed_sfreq = preprocess_joint_eeg(reordered_preview, source_sfreq=raw_sfreq, args=args)

                    fmri_discard_initial_trs = int(args.ds002336_discard_initial_trs)
                    if args.ds002336_fmri_source != "raw":
                        fmri_discard_initial_trs = 0
                    if args.fmri_mode == "roi":
                        fmri_source, _, _ = extract_roi_timeseries(
                            fmri_nii_path=fmri_nii,
                            labels_img=labels_img,
                            tr=args.tr,
                            standardize_fmri=args.standardize_fmri,
                            discard_initial_trs=fmri_discard_initial_trs,
                            include_metadata=True,
                        )
                        total_trs = int(fmri_source.shape[1])
                    else:
                        raw_fmri, _, voxel_size = load_bold_volume(fmri_nii, discard_initial_trs=fmri_discard_initial_trs, include_metadata=True)
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
                        sample_id = f"ds002336_{canonical_subject}_{task}_tr-{tr_index:04d}"
                        record_template = ContrastiveSampleRecord(
                            sample_id=sample_id,
                            dataset="ds002336",
                            subject=canonical_subject,
                            subject_uid=subject_uid,
                            original_subject=original_subject,
                            task=task,
                            run=task,
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
                            records=records,
                        )
                        exported_pairs += 1

                    summaries.append(
                        ContrastiveRunSummary(
                            dataset="ds002336",
                            subject=canonical_subject,
                            subject_uid=subject_uid,
                            original_subject=original_subject,
                            task=task,
                            run=task,
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
                    dataset="ds002336",
                    canonical_subject=canonical_subject,
                    original_subject=original_subject,
                    training_ready=bool(args.training_ready),
                )
                if subject_record is not None:
                    subject_records.append(subject_record)

        if dataset_name == "ds002739":
            ds_root = args.ds002739_root
            electrode_template = load_electrode_template(ds_root)
            for original_subject in tqdm(dataset_subjects["ds002739"], desc="Preparing ds002739 joint contrastive"):
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
                    bold_path = func_dir / f"{original_subject}_task-main_{run}_bold.nii.gz"
                    fmri_events_path = func_dir / f"{original_subject}_task-main_{run}_events.tsv"
                    eeg_data_path = eeg_dir / f"EEG_data_{original_subject}_{run}.mat"
                    eeg_events_path = eeg_dir / f"EEG_events_{original_subject}_{run}.mat"
                    if not all(path.exists() for path in [bold_path, fmri_events_path, eeg_data_path, eeg_events_path]):
                        continue

                    raw_eeg, raw_sfreq, kept_electrodes = load_eeg_data(eeg_data_path, electrode_template=electrode_template)
                    if not any(row.get("dataset") == "ds002739" and "source_channel_name" in row for row in channel_mapping_rows):
                        reordered_preview, mapping_rows = reorder_eeg_channels(raw_eeg, kept_electrodes, target_channel_names)
                        channel_mapping_rows.extend({"dataset": "ds002739", **row} for row in mapping_rows)
                    else:
                        reordered_preview, _ = reorder_eeg_channels(raw_eeg, kept_electrodes, target_channel_names)
                    eeg_data, processed_sfreq = preprocess_joint_eeg(reordered_preview, source_sfreq=raw_sfreq, args=args)

                    eeg_events = load_eeg_events(eeg_events_path)
                    eeg_trials = build_eeg_trial_table(eeg_events)
                    fmri_events = load_fmri_events(fmri_events_path, event_type=args.ds002739_fmri_event_type)
                    if eeg_trials.empty or fmri_events.empty:
                        continue
                    pair_count = min(len(eeg_trials), len(fmri_events))
                    eeg_trials = eeg_trials.iloc[:pair_count].reset_index(drop=True)
                    fmri_events = fmri_events.iloc[:pair_count].reset_index(drop=True)
                    eeg_fmri_offset_sec = float(np.median(
                        eeg_trials["eeg_onset_sec"].to_numpy(dtype=np.float64) - fmri_events["onset"].to_numpy(dtype=np.float64)
                    ))

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
                        eeg_window_end_sec = anchor_sec + eeg_fmri_offset_sec
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
                            records=records,
                        )
                        exported_pairs += 1

                    summaries.append(
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
                            eeg_fmri_offset_sec=eeg_fmri_offset_sec,
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
                if subject_record is not None:
                    subject_records.append(subject_record)

    if not records and not subject_records:
        raise RuntimeError("No joint contrastive samples were exported. Check dataset roots, selected subjects, and alignment settings.")

    if subject_records:
        pd.DataFrame(record.__dict__ for record in sorted(subject_records, key=lambda item: item.subject_uid)).to_csv(out_root / "manifest_all.csv", index=False)
    else:
        pd.DataFrame(record.__dict__ for record in sorted(records, key=lambda item: item.sample_id)).to_csv(out_root / "manifest_all.csv", index=False)
    pd.DataFrame(summary.__dict__ for summary in sorted(summaries, key=lambda item: (item.dataset, item.subject_uid, item.run))).to_csv(out_root / "run_summary.csv", index=False)
    write_subject_mapping(subject_mapping_rows, out_root / "subject_mapping.csv")
    write_channel_metadata(dataset_channel_rows, out_root / "eeg_channels_dataset.csv")
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
    write_channel_metadata(channel_mapping_rows, out_root / "eeg_channel_mapping.csv")

if __name__ == "__main__":
    main()