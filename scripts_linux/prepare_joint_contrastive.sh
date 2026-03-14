#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DS002336_ROOT="../ds002336"
DS002338_ROOT="../ds002338"
DS002739_ROOT="../ds002739"
OUTPUT_ROOT="cache/joint_contrastive"
DATASETS=("ds002336" "ds002739" "ds002338")
DS002336_SUBJECTS=()
DS002338_SUBJECTS=()
DS002739_SUBJECTS=()
DS002336_TASKS=("motorloc" "MIpre" "MIpost" "eegNF" "fmriNF" "eegfmriNF")
DS002338_TASKS=("MIpre" "MIpost" "1dNF" "2dNF")
DS002739_RUNS=()
EEG_WINDOW_SEC="8.0"
TRAINING_READY="true"
SKIP_EXISTING_DATASETS="true"
NUM_WORKERS="2"
FORCE_REFRESH_DATASETS=()
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
        --ds002336-root) DS002336_ROOT="$2"; shift 2 ;;
        --ds002338-root) DS002338_ROOT="$2"; shift 2 ;;
        --ds002739-root) DS002739_ROOT="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --datasets) mapfile -t DATASETS < <(parse_array_arg "$2"); shift 2 ;;
        --ds002336-subjects) mapfile -t DS002336_SUBJECTS < <(parse_array_arg "$2"); shift 2 ;;
        --ds002338-subjects) mapfile -t DS002338_SUBJECTS < <(parse_array_arg "$2"); shift 2 ;;
        --ds002739-subjects) mapfile -t DS002739_SUBJECTS < <(parse_array_arg "$2"); shift 2 ;;
        --ds002336-tasks) mapfile -t DS002336_TASKS < <(parse_array_arg "$2"); shift 2 ;;
        --ds002338-tasks) mapfile -t DS002338_TASKS < <(parse_array_arg "$2"); shift 2 ;;
        --ds002739-runs) mapfile -t DS002739_RUNS < <(parse_array_arg "$2"); shift 2 ;;
        --eeg-window-sec) EEG_WINDOW_SEC="$2"; shift 2 ;;
        --training-ready) TRAINING_READY="true"; shift ;;
        --no-training-ready) TRAINING_READY="false"; shift ;;
        --skip-existing-datasets) SKIP_EXISTING_DATASETS="true"; shift ;;
        --no-skip-existing-datasets) SKIP_EXISTING_DATASETS="false"; shift ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --force-refresh-datasets) mapfile -t FORCE_REFRESH_DATASETS < <(parse_array_arg "$2"); shift 2 ;;
        --python-exe) PYTHON_EXE="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
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
    "${REPO_ROOT}/preprocess/prepare_joint_contrastive.py"
    "--output-root" "${OUTPUT_ROOT}"
    "--eeg-window-sec" "${EEG_WINDOW_SEC}"
    "--fmri-mode" "volume"
    "--fmri-voxel-size" "2.0" "2.0" "2.0"
    "--fmri-max-shape" "48" "48" "48"
    "--tr" "2.0"
    "--eeg-mode" "patched"
    "--eeg-target-sfreq" "200"
    "--eeg-lfreq" "0.5"
    "--eeg-hfreq" "40"
    "--num-workers" "${NUM_WORKERS}"
    "--pack-subject-files"
)

if [[ ${#DATASETS[@]} -gt 0 ]]; then
    CLI_ARGS+=("--datasets" "${DATASETS[@]}")
fi

contains_item() {
    local needle="$1"
    shift
    for item in "$@"; do
        if [[ "${item}" == "${needle}" ]]; then
            return 0
        fi
    done
    return 1
}

if contains_item "ds002336" "${DATASETS[@]}"; then
    CLI_ARGS+=("--ds002336-root" "${DS002336_ROOT}")
    if [[ ${#DS002336_SUBJECTS[@]} -gt 0 ]]; then
        CLI_ARGS+=("--ds002336-subjects" "${DS002336_SUBJECTS[@]}")
    fi
    if [[ ${#DS002336_TASKS[@]} -gt 0 ]]; then
        CLI_ARGS+=("--ds002336-tasks" "${DS002336_TASKS[@]}")
    fi
    CLI_ARGS+=("--ds002336-drop-ecg")
fi

if contains_item "ds002338" "${DATASETS[@]}"; then
    CLI_ARGS+=("--ds002338-root" "${DS002338_ROOT}")
    if [[ ${#DS002338_SUBJECTS[@]} -gt 0 ]]; then
        CLI_ARGS+=("--ds002338-subjects" "${DS002338_SUBJECTS[@]}")
    fi
    if [[ ${#DS002338_TASKS[@]} -gt 0 ]]; then
        CLI_ARGS+=("--ds002338-tasks" "${DS002338_TASKS[@]}")
    fi
    CLI_ARGS+=("--ds002338-drop-ecg")
fi

if contains_item "ds002739" "${DATASETS[@]}"; then
    CLI_ARGS+=("--ds002739-root" "${DS002739_ROOT}")
    if [[ ${#DS002739_SUBJECTS[@]} -gt 0 ]]; then
        CLI_ARGS+=("--ds002739-subjects" "${DS002739_SUBJECTS[@]}")
    fi
    if [[ ${#DS002739_RUNS[@]} -gt 0 ]]; then
        CLI_ARGS+=("--ds002739-runs" "${DS002739_RUNS[@]}")
    fi
fi

if [[ "${TRAINING_READY}" == "true" ]]; then
    CLI_ARGS+=("--training-ready")
else
    CLI_ARGS+=("--no-training-ready")
fi

if [[ "${SKIP_EXISTING_DATASETS}" == "true" ]]; then
    CLI_ARGS+=("--skip-existing-datasets")
fi

if [[ ${#FORCE_REFRESH_DATASETS[@]} -gt 0 ]]; then
    CLI_ARGS+=("--force-refresh-datasets" "${FORCE_REFRESH_DATASETS[@]}")
fi

echo "Preparing joint contrastive cache..."
echo "Datasets: ${DATASETS[*]}"
echo "Output root: ${OUTPUT_ROOT}"
echo "EEG window sec: ${EEG_WINDOW_SEC}"

"${PYTHON}" "${CLI_ARGS[@]}"
