#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DS_ROOT="../SEED"
OUTPUT_ROOT="cache/ds009999"
LABELS_MAT=""
SUBJECTS=()
SESSIONS=()
SPLIT_MODE="loso"
TRAIN_SUBJECTS="12"
VAL_SUBJECTS="2"
TEST_SUBJECTS="1"
INPUT_SFREQ="200.0"
EEG_TARGET_SFREQ="200.0"
WINDOW_SEC="8.0"
WINDOW_OVERLAP_SEC="0.0"
EEG_SEQ_LEN="8"
EEG_PATCH_LEN="200"
TRAINING_READY="true"
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
        --ds-root) DS_ROOT="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --labels-mat) LABELS_MAT="$2"; shift 2 ;;
        --subjects) mapfile -t SUBJECTS < <(parse_array_arg "$2"); shift 2 ;;
        --sessions) mapfile -t SESSIONS < <(parse_array_arg "$2"); shift 2 ;;
        --split-mode) SPLIT_MODE="$2"; shift 2 ;;
        --train-subjects) TRAIN_SUBJECTS="$2"; shift 2 ;;
        --val-subjects) VAL_SUBJECTS="$2"; shift 2 ;;
        --test-subjects) TEST_SUBJECTS="$2"; shift 2 ;;
        --input-sfreq) INPUT_SFREQ="$2"; shift 2 ;;
        --eeg-target-sfreq) EEG_TARGET_SFREQ="$2"; shift 2 ;;
        --window-sec) WINDOW_SEC="$2"; shift 2 ;;
        --window-overlap-sec) WINDOW_OVERLAP_SEC="$2"; shift 2 ;;
        --eeg-seq-len) EEG_SEQ_LEN="$2"; shift 2 ;;
        --eeg-patch-len) EEG_PATCH_LEN="$2"; shift 2 ;;
        --training-ready) TRAINING_READY="true"; shift ;;
        --no-training-ready) TRAINING_READY="false"; shift ;;
        --target-channel-manifest) TARGET_CHANNEL_MANIFEST="$2"; shift 2 ;;
        --python-exe) PYTHON_EXE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ -n "${PYTHON_EXE}" ]]; then
    PYTHON="${PYTHON_EXE}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
else
    PYTHON="python"
fi

cd "${REPO_ROOT}"

CLI_ARGS=(
    "preprocess/prepare_ds009999.py"
    "--ds-root" "${DS_ROOT}"
    "--output-root" "${OUTPUT_ROOT}"
    "--pack-subject-files"
    "--split-mode" "${SPLIT_MODE}"
    "--train-subjects" "${TRAIN_SUBJECTS}"
    "--val-subjects" "${VAL_SUBJECTS}"
    "--test-subjects" "${TEST_SUBJECTS}"
    "--input-sfreq" "${INPUT_SFREQ}"
    "--eeg-target-sfreq" "${EEG_TARGET_SFREQ}"
    "--window-sec" "${WINDOW_SEC}"
    "--window-overlap-sec" "${WINDOW_OVERLAP_SEC}"
    "--eeg-mode" "patched"
    "--eeg-seq-len" "${EEG_SEQ_LEN}"
    "--eeg-patch-len" "${EEG_PATCH_LEN}"
)

if [[ -n "${LABELS_MAT}" ]]; then
    CLI_ARGS+=("--labels-mat" "${LABELS_MAT}")
fi
if [[ -n "${TARGET_CHANNEL_MANIFEST}" ]]; then
    CLI_ARGS+=("--target-channel-manifest" "${TARGET_CHANNEL_MANIFEST}")
fi
if [[ "${TRAINING_READY}" == "true" ]]; then
    CLI_ARGS+=("--training-ready")
else
    CLI_ARGS+=("--no-training-ready")
fi
if [[ ${#SUBJECTS[@]} -gt 0 ]]; then
    CLI_ARGS+=("--subjects" "${SUBJECTS[@]}")
fi
if [[ ${#SESSIONS[@]} -gt 0 ]]; then
    CLI_ARGS+=("--sessions" "${SESSIONS[@]}")
fi

echo "Preparing ds009999 (SEED) dataset..."
echo "Output root: ${OUTPUT_ROOT}"
echo "Split mode: ${SPLIT_MODE}"

"${PYTHON}" "${CLI_ARGS[@]}"
