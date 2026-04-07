#!/usr/bin/env bash
# ./scripts_linux/run_finetune_from_pretrain_trials.sh --config configs/finetune_ds009999.yaml --pretrain-root ../pretrain_save --output-root outputs/finetune_from_pretrain_trials --gpu-count 1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH=""
PRETRAIN_ROOT="../pretrain_save"
CHECKPOINT_RELPATH="run_output/contrastive/checkpoints/best.pth"
TRIAL_GLOB="trial_*"
OUTPUT_ROOT=""
EPOCHS=""
BATCH_SIZE=""
EVAL_BATCH_SIZE=""
NUM_WORKERS=""
GPU_COUNT="1"
GPU_IDS=""
FORCE_CPU="false"
TEST_ONLY="false"
SKIP_EXISTING="false"
PYTHON_EXE=""
EXTRA_SET_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --pretrain-root) PRETRAIN_ROOT="$2"; shift 2 ;;
        --checkpoint-relpath) CHECKPOINT_RELPATH="$2"; shift 2 ;;
        --trial-glob) TRIAL_GLOB="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --gpu-count) GPU_COUNT="$2"; shift 2 ;;
        --gpu-ids) GPU_IDS="$2"; shift 2 ;;
        --force-cpu) FORCE_CPU="true"; shift ;;
        --test-only) TEST_ONLY="true"; shift ;;
        --skip-existing) SKIP_EXISTING="true"; shift ;;
        --python-exe) PYTHON_EXE="$2"; shift 2 ;;
        --set) EXTRA_SET_ARGS+=("$2"); shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${CONFIG_PATH}" ]]; then
    echo "--config is required" >&2
    exit 2
fi

if [[ -n "${PYTHON_EXE}" ]]; then
    PYTHON="${PYTHON_EXE}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
else
    PYTHON="python"
fi

cd "${REPO_ROOT}"

normalize_path() {
    local raw_path="$1"
    "${PYTHON}" - <<'PY' "${raw_path}"
import os
import sys
print(os.path.normpath(sys.argv[1]))
PY
}

CONFIG_PATH="$(normalize_path "${CONFIG_PATH}")"
PRETRAIN_ROOT="$(normalize_path "${PRETRAIN_ROOT}")"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config not found: ${CONFIG_PATH}" >&2
    exit 1
fi

read_config_value() {
    local config_path="$1"
    local dotted_key="$2"
    "${PYTHON}" - <<'PY' "${config_path}" "${dotted_key}"
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
dotted_key = sys.argv[2]
with config_path.open("r", encoding="utf-8") as handle:
    payload = yaml.safe_load(handle) or {}
cursor = payload
for part in dotted_key.split("."):
    if not isinstance(cursor, dict):
        cursor = ""
        break
    cursor = cursor.get(part, "")
if cursor is None:
    cursor = ""
print(cursor)
PY
}

if [[ -z "${OUTPUT_ROOT}" ]]; then
    CONFIG_STEM="$(basename "${CONFIG_PATH}" .yaml)"
    OUTPUT_ROOT="outputs/${CONFIG_STEM}_from_pretrain_trials"
fi

mkdir -p "${OUTPUT_ROOT}"

ROOT_DIR="$(read_config_value "${CONFIG_PATH}" "data.root_dir")"
if [[ -z "${ROOT_DIR}" ]]; then
    echo "data.root_dir is empty in ${CONFIG_PATH}" >&2
    exit 1
fi

if [[ ! -d "${ROOT_DIR}" ]]; then
    echo "data.root_dir does not exist: ${ROOT_DIR}" >&2
    exit 1
fi

if [[ ! -d "${ROOT_DIR}/loso_subjectwise" ]]; then
    echo "LOSO directory not found: ${ROOT_DIR}/loso_subjectwise" >&2
    exit 1
fi

write_trial_summary() {
    local trial_root="$1"
    "${PYTHON}" - <<'PY' "${trial_root}"
import csv
import json
import os
import statistics
import sys

root = sys.argv[1]
rows = []
for name in sorted(os.listdir(root)):
    fold_dir = os.path.join(root, name)
    if not os.path.isdir(fold_dir):
        continue
    metrics_path = os.path.join(fold_dir, "test_metrics.json")
    if not os.path.isfile(metrics_path):
        metrics_path = os.path.join(fold_dir, "final_metrics.json")
    if not os.path.isfile(metrics_path):
        continue
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle) or {}
    source = metrics.get("test_metrics") if isinstance(metrics, dict) and isinstance(metrics.get("test_metrics"), dict) else metrics
    rows.append({
        "fold": name,
        "accuracy": float(source.get("accuracy", 0.0) or 0.0),
        "macro_f1": float(source.get("macro_f1", 0.0) or 0.0),
        "loss": float(source.get("loss", 0.0) or 0.0),
    })

if not rows:
    raise SystemExit(0)

def mean(values):
    return statistics.mean(values) if values else 0.0

def std(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0

summary = {
    "fold": "MEAN_STD",
    "accuracy": mean([row["accuracy"] for row in rows]),
    "macro_f1": mean([row["macro_f1"] for row in rows]),
    "loss": mean([row["loss"] for row in rows]),
    "accuracy_std": std([row["accuracy"] for row in rows]),
    "macro_f1_std": std([row["macro_f1"] for row in rows]),
    "loss_std": std([row["loss"] for row in rows]),
}
for row in rows:
    row["accuracy_std"] = ""
    row["macro_f1_std"] = ""
    row["loss_std"] = ""
rows.append(summary)

output_csv = os.path.join(root, "trial_finetune_summary.csv")
with open(output_csv, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["fold", "accuracy", "accuracy_std", "macro_f1", "macro_f1_std", "loss", "loss_std"],
    )
    writer.writeheader()
    writer.writerows(rows)
print(output_csv)
PY
}

write_global_summary() {
    local output_root="$1"
    "${PYTHON}" - <<'PY' "${output_root}"
import csv
import os
import sys

root = sys.argv[1]
rows = []
for trial_name in sorted(os.listdir(root)):
    trial_dir = os.path.join(root, trial_name)
    if not os.path.isdir(trial_dir):
        continue
    summary_path = os.path.join(trial_dir, "trial_finetune_summary.csv")
    if not os.path.isfile(summary_path):
        continue
    with open(summary_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("fold") == "MEAN_STD":
                rows.append({
                    "trial": trial_name,
                    "accuracy_mean": row.get("accuracy", ""),
                    "accuracy_std": row.get("accuracy_std", ""),
                    "macro_f1_mean": row.get("macro_f1", ""),
                    "macro_f1_std": row.get("macro_f1_std", ""),
                    "loss_mean": row.get("loss", ""),
                    "loss_std": row.get("loss_std", ""),
                })
                break

if not rows:
    raise SystemExit(0)

output_csv = os.path.join(root, "all_trials_finetune_summary.csv")
with open(output_csv, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["trial", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std", "loss_mean", "loss_std"],
    )
    writer.writeheader()
    writer.writerows(rows)
print(output_csv)
PY
}

if [[ "${FORCE_CPU}" != "true" && "${GPU_COUNT}" -le 0 ]]; then
    echo "--gpu-count must be >= 1 unless --force-cpu is used" >&2
    exit 2
fi

IFS=',' read -r -a GPU_ID_LIST <<< "${GPU_IDS}"
NONEMPTY_GPU_IDS=()
for item in "${GPU_ID_LIST[@]}"; do
    trimmed="$(echo "${item}" | xargs)"
    if [[ -n "${trimmed}" ]]; then
        NONEMPTY_GPU_IDS+=("${trimmed}")
    fi
done
if [[ ${#NONEMPTY_GPU_IDS[@]} -gt 0 && "${FORCE_CPU}" != "true" ]]; then
    export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${NONEMPTY_GPU_IDS[*]}")"
    echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

USE_MULTI_GPU="false"
if [[ "${FORCE_CPU}" != "true" && "${GPU_COUNT}" -gt 1 ]]; then
    USE_MULTI_GPU="true"
fi

shopt -s nullglob
CHECKPOINT_PATHS=("${PRETRAIN_ROOT}"/${TRIAL_GLOB}/${CHECKPOINT_RELPATH})
shopt -u nullglob

if [[ ${#CHECKPOINT_PATHS[@]} -eq 0 ]]; then
    echo "No checkpoints found under ${PRETRAIN_ROOT}/${TRIAL_GLOB}/${CHECKPOINT_RELPATH}" >&2
    exit 1
fi

echo "Found ${#CHECKPOINT_PATHS[@]} pretrain checkpoints."

run_one_finetune() {
    local checkpoint_path="$1"
    local trial_output_root="$2"
    local fold_name="$3"
    local train_manifest="$4"
    local val_manifest="$5"
    local test_manifest="$6"

    local fold_output_dir="${trial_output_root}/${fold_name}"
    mkdir -p "${fold_output_dir}"

    local args=(
        "--config" "${CONFIG_PATH}"
        "--train-manifest" "${train_manifest}"
        "--val-manifest" "${val_manifest}"
        "--test-manifest" "${test_manifest}"
        "--root-dir" "${ROOT_DIR}"
        "--output-dir" "${fold_output_dir}"
        "--contrastive-checkpoint" "${checkpoint_path}"
    )

    if [[ -n "${EPOCHS}" ]]; then
        args+=("--epochs" "${EPOCHS}")
    fi
    if [[ -n "${BATCH_SIZE}" ]]; then
        args+=("--batch-size" "${BATCH_SIZE}")
    fi
    if [[ -n "${EVAL_BATCH_SIZE}" ]]; then
        args+=("--eval-batch-size" "${EVAL_BATCH_SIZE}")
    fi
    if [[ -n "${NUM_WORKERS}" ]]; then
        args+=("--num-workers" "${NUM_WORKERS}")
    fi
    if [[ "${FORCE_CPU}" == "true" ]]; then
        args+=("--force-cpu")
    else
        args+=("--set" "train.gpu_count=${GPU_COUNT}")
        if [[ ${#NONEMPTY_GPU_IDS[@]} -gt 0 ]]; then
            args+=("--set" "train.gpu_ids=$(IFS=,; echo "${NONEMPTY_GPU_IDS[*]}")")
        fi
    fi
    if [[ "${TEST_ONLY}" == "true" ]]; then
        local eval_checkpoint="${fold_output_dir}/checkpoints/best.pth"
        if [[ ! -f "${eval_checkpoint}" ]]; then
            echo "Missing finetune checkpoint for --test-only: ${eval_checkpoint}" >&2
            exit 1
        fi
        args+=("--finetune-checkpoint" "${eval_checkpoint}" "--test-only")
    fi
    for extra_set in "${EXTRA_SET_ARGS[@]}"; do
        args+=("--set" "${extra_set}")
    done

    if [[ "${USE_MULTI_GPU}" == "true" ]]; then
        "${PYTHON}" -m torch.distributed.run --nproc_per_node "${GPU_COUNT}" "${REPO_ROOT}/run_finetune.py" "${args[@]}"
    else
        "${PYTHON}" "${REPO_ROOT}/run_finetune.py" "${args[@]}"
    fi
}

for checkpoint_path in "${CHECKPOINT_PATHS[@]}"; do
    trial_dir="$(dirname "$(dirname "$(dirname "$(dirname "${checkpoint_path}")")")")"
    trial_name="$(basename "${trial_dir}")"
    trial_output_root="${OUTPUT_ROOT}/${trial_name}"

    if [[ "${SKIP_EXISTING}" == "true" && -f "${trial_output_root}/trial_finetune_summary.csv" ]]; then
        echo "[${trial_name}] skip existing"
        continue
    fi

    mkdir -p "${trial_output_root}"
    echo "[${trial_name}] checkpoint=${checkpoint_path}"

    shopt -s nullglob
    FOLD_DIRS=("${ROOT_DIR}"/loso_subjectwise/fold_*)
    shopt -u nullglob
    if [[ ${#FOLD_DIRS[@]} -eq 0 ]]; then
        echo "No LOSO folds found under ${ROOT_DIR}/loso_subjectwise" >&2
        exit 1
    fi
    for fold_dir in "${FOLD_DIRS[@]}"; do
        fold_name="$(basename "${fold_dir}")"
        echo "  [${trial_name}/${fold_name}] finetune"
        run_one_finetune \
            "${checkpoint_path}" \
            "${trial_output_root}" \
            "${fold_name}" \
            "${fold_dir}/manifest_train.csv" \
            "${fold_dir}/manifest_val.csv" \
            "${fold_dir}/manifest_test.csv"
    done

    write_trial_summary "${trial_output_root}" >/dev/null || true
done

write_global_summary "${OUTPUT_ROOT}" >/dev/null || true
echo "Wrote finetune outputs under ${OUTPUT_ROOT}"
