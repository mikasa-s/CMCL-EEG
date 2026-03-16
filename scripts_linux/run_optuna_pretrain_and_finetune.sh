#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRETRAIN_DATASETS=("ds002336" "ds002338" "ds002739")
TARGET_DATASET="ds002739"
JOINT_TRAIN_CONFIG="configs/train_joint_contrastive.yaml"
FINETUNE_CONFIG="configs/finetune_ds002739.yaml"
OUTPUT_ROOT="outputs/optuna_run"
CACHE_ROOT="cache"
JOINT_EEG_WINDOW_SEC="8.0"
PRETRAIN_EPOCHS="0"
FINETUNE_EPOCHS="0"
PRETRAIN_BATCH_SIZE="0"
FINETUNE_BATCH_SIZE="0"
BATCH_SIZE="0"
EVAL_BATCH_SIZE="0"
NUM_WORKERS="-1"
GPU_COUNT="1"
GPU_IDS=""
SKIP_PRETRAIN="false"
SKIP_FINETUNE="false"
TEST_ONLY="false"
FORCE_CPU="false"
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
        --pretrain-datasets) mapfile -t PRETRAIN_DATASETS < <(parse_array_arg "$2"); shift 2 ;;
        --target-dataset) TARGET_DATASET="$2"; shift 2 ;;
        --joint-train-config) JOINT_TRAIN_CONFIG="$2"; shift 2 ;;
        --finetune-config) FINETUNE_CONFIG="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --cache-root) CACHE_ROOT="$2"; shift 2 ;;
        --joint-eeg-window-sec) JOINT_EEG_WINDOW_SEC="$2"; shift 2 ;;
        --pretrain-epochs) PRETRAIN_EPOCHS="$2"; shift 2 ;;
        --finetune-epochs) FINETUNE_EPOCHS="$2"; shift 2 ;;
        --pretrain-batch-size) PRETRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --finetune-batch-size) FINETUNE_BATCH_SIZE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --gpu-count) GPU_COUNT="$2"; shift 2 ;;
        --gpu-ids) GPU_IDS="$2"; shift 2 ;;
        --skip-pretrain) SKIP_PRETRAIN="true"; shift ;;
        --skip-finetune) SKIP_FINETUNE="true"; shift ;;
        --test-only) TEST_ONLY="true"; shift ;;
        --force-cpu) FORCE_CPU="true"; shift ;;
        --python-exe) PYTHON_EXE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

cd "${REPO_ROOT}"

if [[ "${OUTPUT_ROOT}" = /* ]]; then
    resolved_output_root="${OUTPUT_ROOT}"
else
    resolved_output_root="$(cd . && pwd)/${OUTPUT_ROOT}"
fi

if [[ "${CACHE_ROOT}" = /* ]]; then
    resolved_cache_root="${CACHE_ROOT}"
else
    resolved_cache_root="$(cd . && pwd)/${CACHE_ROOT}"
fi

joint_cache_root="${resolved_cache_root}/joint_contrastive"
ds002336_cache_root="${resolved_cache_root}/ds002336"
ds002338_cache_root="${resolved_cache_root}/ds002338"
ds002739_cache_root="${resolved_cache_root}/ds002739"
shared_output_root="${resolved_output_root}"

args=(
    "${REPO_ROOT}/scripts_linux/run_pretrain_and_finetune.sh"
    "--python-exe" "${PYTHON_EXE}"
    "--pretrain-datasets" "$(IFS=,; echo "${PRETRAIN_DATASETS[*]}")"
    "--target-dataset" "${TARGET_DATASET}"
    "--joint-train-config" "${JOINT_TRAIN_CONFIG}"
    "--joint-cache-root" "${joint_cache_root}"
    "--ds002336-cache-root" "${ds002336_cache_root}"
    "--ds002338-cache-root" "${ds002338_cache_root}"
    "--ds002739-cache-root" "${ds002739_cache_root}"
    "--joint-output-root" "${shared_output_root}"
    "--ds002336-output-root" "${shared_output_root}"
    "--ds002338-output-root" "${shared_output_root}"
    "--ds002739-output-root" "${shared_output_root}"
    "--joint-eeg-window-sec" "${JOINT_EEG_WINDOW_SEC}"
)

if [[ "${TARGET_DATASET}" == "ds002336" ]]; then
    args+=("--ds002336-finetune-config" "${FINETUNE_CONFIG}")
elif [[ "${TARGET_DATASET}" == "ds002338" ]]; then
    args+=("--ds002338-finetune-config" "${FINETUNE_CONFIG}")
else
    args+=("--ds002739-finetune-config" "${FINETUNE_CONFIG}")
fi

if [[ ${PRETRAIN_EPOCHS} -gt 0 ]]; then args+=("--pretrain-epochs" "${PRETRAIN_EPOCHS}"); fi
if [[ ${FINETUNE_EPOCHS} -gt 0 ]]; then args+=("--finetune-epochs" "${FINETUNE_EPOCHS}"); fi
if [[ ${BATCH_SIZE} -gt 0 ]]; then args+=("--batch-size" "${BATCH_SIZE}"); fi
if [[ ${PRETRAIN_BATCH_SIZE} -gt 0 ]]; then args+=("--pretrain-batch-size" "${PRETRAIN_BATCH_SIZE}"); fi
if [[ ${FINETUNE_BATCH_SIZE} -gt 0 ]]; then args+=("--finetune-batch-size" "${FINETUNE_BATCH_SIZE}"); fi
if [[ ${EVAL_BATCH_SIZE} -gt 0 ]]; then args+=("--eval-batch-size" "${EVAL_BATCH_SIZE}"); fi
if [[ ${NUM_WORKERS} -ge 0 ]]; then args+=("--num-workers" "${NUM_WORKERS}"); fi
if [[ ${GPU_COUNT} -gt 0 ]]; then args+=("--gpu-count" "${GPU_COUNT}"); fi
if [[ -n "${GPU_IDS}" ]]; then args+=("--gpu-ids" "${GPU_IDS}"); fi
if [[ "${SKIP_PRETRAIN}" == "true" ]]; then args+=("--skip-pretrain"); fi
if [[ "${SKIP_FINETUNE}" == "true" ]]; then args+=("--skip-finetune"); fi
if [[ "${TEST_ONLY}" == "true" ]]; then args+=("--test-only"); fi
if [[ "${FORCE_CPU}" == "true" ]]; then args+=("--force-cpu"); fi

"${args[@]}"
