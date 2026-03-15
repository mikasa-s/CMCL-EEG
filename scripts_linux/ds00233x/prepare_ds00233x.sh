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
BLOCK_WINDOW_SEC="8.0"
BLOCK_OVERLAP_SEC="2.0"
DROP_ECG="true"
TRAINING_READY="true"
EEG_ONLY="true"
TARGET_CHANNEL_MANIFEST=""
FMRI_SOURCE="spm_smoothed"
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
        --block-window-sec) BLOCK_WINDOW_SEC="$2"; shift 2 ;;
        --block-overlap-sec) BLOCK_OVERLAP_SEC="$2"; shift 2 ;;
        --drop-ecg) DROP_ECG="true"; shift ;;
        --no-drop-ecg) DROP_ECG="false"; shift ;;
        --training-ready) TRAINING_READY="true"; shift ;;
        --no-training-ready) TRAINING_READY="false"; shift ;;
        --eeg-only) EEG_ONLY="true"; shift ;;
        --no-eeg-only) EEG_ONLY="false"; shift ;;
        --target-channel-manifest) TARGET_CHANNEL_MANIFEST="$2"; shift 2 ;;
        --fmri-source) FMRI_SOURCE="$2"; shift 2 ;;
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

if [[ ! -d "${DS_ROOT}" ]]; then
    candidates=(
        "data/${DATASET_NAME}"
        "../data/${DATASET_NAME}"
        "../${DATASET_NAME}"
    )
    for candidate in "${candidates[@]}"; do
        if [[ -d "${candidate}" ]]; then
            DS_ROOT="${candidate}"
            break
        fi
    done
fi

if [[ ${#TASKS[@]} -eq 0 ]]; then
    if [[ "${DATASET_NAME}" == "ds002336" ]]; then
        TASKS=("motorloc" "MIpre" "MIpost" "eegNF" "fmriNF" "eegfmriNF")
    elif [[ "${DATASET_NAME}" == "ds002338" ]]; then
        TASKS=("MIpre" "MIpost" "1dNF" "2dNF")
    fi
fi

# Keep parity with current PowerShell behavior.
if [[ "${DATASET_NAME}" == "ds002336" ]]; then
    TRAIN_SUBJECTS="7"
    VAL_SUBJECTS="2"
    TEST_SUBJECTS="1"
elif [[ "${DATASET_NAME}" == "ds002338" ]]; then
    TRAIN_SUBJECTS="14"
    VAL_SUBJECTS="2"
    TEST_SUBJECTS="1"
fi

if [[ "${SPLIT_MODE}" != "none" && ${#SUBJECTS[@]} -gt 0 ]]; then
    if [[ "${SPLIT_MODE}" == "loso" ]]; then
        required_subjects=$((VAL_SUBJECTS + 1))
    else
        required_subjects=$((TRAIN_SUBJECTS + VAL_SUBJECTS + TEST_SUBJECTS))
    fi
    if [[ ${#SUBJECTS[@]} -lt ${required_subjects} ]]; then
        echo "Provided subject subset is smaller than the requested split sizes; disabling split generation for this run." >&2
        SPLIT_MODE="none"
    fi
fi

CLI_ARGS=(
    "preprocess/prepare_ds00233x.py"
    "--ds-root" "${DS_ROOT}"
    "--output-root" "${OUTPUT_ROOT}"
    "--sample-mode" "block"
    "--label-mode" "binary_rest_task"
    "--fmri-mode" "volume"
    "--pack-subject-files"
    "--eeg-mode" "patched"
    "--eeg-seq-len" "${EEG_SEQ_LEN}"
    "--eeg-patch-len" "${EEG_PATCH_LEN}"
    "--block-window-sec" "${BLOCK_WINDOW_SEC}"
    "--block-overlap-sec" "${BLOCK_OVERLAP_SEC}"
    "--tr" "2.0"
    "--fmri-max-shape" "48" "48" "48"
    "--split-mode" "${SPLIT_MODE}"
    "--train-subjects" "${TRAIN_SUBJECTS}"
    "--val-subjects" "${VAL_SUBJECTS}"
    "--test-subjects" "${TEST_SUBJECTS}"
    "--dataset-name" "${DATASET_NAME}"
)

if [[ "${FMRI_SOURCE}" != "raw" ]]; then
    CLI_ARGS+=("--fmri-source" "${FMRI_SOURCE}" "--discard-initial-trs" "0" "--protocol-offset-sec" "0.0")
else
    discard_initial_trs="1"
    if [[ "${DATASET_NAME}" == "ds002338" ]]; then
        discard_initial_trs="2"
    fi
    CLI_ARGS+=("--discard-initial-trs" "${discard_initial_trs}" "--protocol-offset-sec" "2.0")
fi

CLI_ARGS+=("--tasks" "${TASKS[@]}")

if [[ "${DROP_ECG}" == "true" ]]; then
    CLI_ARGS+=("--drop-ecg")
fi
if [[ "${TRAINING_READY}" == "true" ]]; then
    CLI_ARGS+=("--training-ready")
else
    CLI_ARGS+=("--no-training-ready")
fi
if [[ "${EEG_ONLY}" == "true" ]]; then
    CLI_ARGS+=("--eeg-only")
fi
if [[ -n "${TARGET_CHANNEL_MANIFEST}" ]]; then
    CLI_ARGS+=("--target-channel-manifest" "${TARGET_CHANNEL_MANIFEST}")
fi
if [[ ${#SUBJECTS[@]} -gt 0 ]]; then
    CLI_ARGS+=("--subjects" "${SUBJECTS[@]}")
fi

echo "Preparing ${DATASET_NAME} dataset..."
echo "Output root: ${OUTPUT_ROOT}"
echo "Split mode: ${SPLIT_MODE}"
echo "fMRI source: ${FMRI_SOURCE}"
echo "EEG-only: ${EEG_ONLY}"

"${PYTHON}" "${CLI_ARGS[@]}"
