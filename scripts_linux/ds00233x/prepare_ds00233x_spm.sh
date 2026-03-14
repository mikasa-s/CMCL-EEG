#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET_NAME="ds002336"
DS_ROOT="../ds002336"
OUTPUT_ROOT="cache/ds002336"
SUBJECTS=()
TASKS=()
SPLIT_MODE="loso"
TRAIN_SUBJECTS="14"
VAL_SUBJECTS="2"
TEST_SUBJECTS="1"
EEG_SEQ_LEN="8"
EEG_PATCH_LEN="200"
DROP_ECG="true"
TRAINING_READY="true"
EEG_ONLY="true"
PARALLEL_WORKERS="4"
TARGET_CHANNEL_MANIFEST=""
PYTHON_EXE=""

parse_array_arg() {
    local raw="$1"
    IFS=',' read -r -a _arr <<< "$raw"
    for _item in "${_arr[@]}"; do
        if [[ -n "${_item}" ]]; then
            echo "${_item}"
        fi
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-name) DATASET_NAME="$2"; shift 2 ;;
        --ds-root) DS_ROOT="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --subjects) mapfile -t SUBJECTS < <(parse_array_arg "$2"); shift 2 ;;
        --tasks) mapfile -t TASKS < <(parse_array_arg "$2"); shift 2 ;;
        --split-mode) SPLIT_MODE="$2"; shift 2 ;;
        --train-subjects) TRAIN_SUBJECTS="$2"; shift 2 ;;
        --val-subjects) VAL_SUBJECTS="$2"; shift 2 ;;
        --test-subjects) TEST_SUBJECTS="$2"; shift 2 ;;
        --eeg-seq-len) EEG_SEQ_LEN="$2"; shift 2 ;;
        --eeg-patch-len) EEG_PATCH_LEN="$2"; shift 2 ;;
        --drop-ecg) DROP_ECG="true"; shift ;;
        --no-drop-ecg) DROP_ECG="false"; shift ;;
        --training-ready) TRAINING_READY="true"; shift ;;
        --no-training-ready) TRAINING_READY="false"; shift ;;
        --eeg-only) EEG_ONLY="true"; shift ;;
        --no-eeg-only) EEG_ONLY="false"; shift ;;
        --parallel-workers) PARALLEL_WORKERS="$2"; shift 2 ;;
        --target-channel-manifest) TARGET_CHANNEL_MANIFEST="$2"; shift 2 ;;
        --python-exe) PYTHON_EXE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ "${PARALLEL_WORKERS}" -lt 1 || "${PARALLEL_WORKERS}" -gt 64 ]]; then
    echo "--parallel-workers must be in [1, 64]" >&2
    exit 2
fi

cd "${REPO_ROOT}"
MATLAB_SCRIPT_DIR="${REPO_ROOT}/preprocess"
RESOLVED_DS_ROOT="$(cd "${DS_ROOT}" && pwd)"

if [[ ${#TASKS[@]} -eq 0 ]]; then
    if [[ "${DATASET_NAME}" == "ds002336" ]]; then
        TASKS=("motorloc" "MIpre" "MIpost" "eegNF" "fmriNF" "eegfmriNF")
    elif [[ "${DATASET_NAME}" == "ds002338" ]]; then
        TASKS=("MIpre" "MIpost" "1dNF" "2dNF")
    fi
fi

SUBJECTS_EXPR="{}"
if [[ ${#SUBJECTS[@]} -gt 0 ]]; then
    SUBJECTS_EXPR="{"$(printf "'%s'," "${SUBJECTS[@]}")"}"
    SUBJECTS_EXPR="${SUBJECTS_EXPR/,}/}"
fi

TASKS_WITH_PREFIX=()
for t in "${TASKS[@]}"; do
    TASKS_WITH_PREFIX+=("task-${t}")
done
TASKS_EXPR="{"$(printf "'%s'," "${TASKS_WITH_PREFIX[@]}")"}"
TASKS_EXPR="${TASKS_EXPR/,}/}"

MATLAB_CMD="addpath('${MATLAB_SCRIPT_DIR}'); run_spm_preproc_ds00233x(${SUBJECTS_EXPR},${TASKS_EXPR},'${RESOLVED_DS_ROOT}',${PARALLEL_WORKERS});"

echo "Running MATLAB SPM preprocessing for ${DATASET_NAME}..."
matlab -batch "${MATLAB_CMD}"

echo "Running Python preprocessing for ${DATASET_NAME}..."
PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_ds00233x.sh"
ARGS=(
    --dataset-name "${DATASET_NAME}"
    --ds-root "${DS_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --split-mode "${SPLIT_MODE}"
    --train-subjects "${TRAIN_SUBJECTS}"
    --val-subjects "${VAL_SUBJECTS}"
    --test-subjects "${TEST_SUBJECTS}"
    --eeg-seq-len "${EEG_SEQ_LEN}"
    --eeg-patch-len "${EEG_PATCH_LEN}"
    --fmri-source "spm_smoothed"
)

if [[ ${#TASKS[@]} -gt 0 ]]; then
    ARGS+=(--tasks "$(IFS=,; echo "${TASKS[*]}")")
fi
if [[ ${#SUBJECTS[@]} -gt 0 ]]; then
    ARGS+=(--subjects "$(IFS=,; echo "${SUBJECTS[*]}")")
fi
if [[ "${DROP_ECG}" == "true" ]]; then ARGS+=(--drop-ecg); else ARGS+=(--no-drop-ecg); fi
if [[ "${TRAINING_READY}" == "true" ]]; then ARGS+=(--training-ready); else ARGS+=(--no-training-ready); fi
if [[ "${EEG_ONLY}" == "true" ]]; then ARGS+=(--eeg-only); else ARGS+=(--no-eeg-only); fi
if [[ -n "${TARGET_CHANNEL_MANIFEST}" ]]; then ARGS+=(--target-channel-manifest "${TARGET_CHANNEL_MANIFEST}"); fi
if [[ -n "${PYTHON_EXE}" ]]; then ARGS+=(--python-exe "${PYTHON_EXE}"); fi

"${PREPARE_SCRIPT}" "${ARGS[@]}"
