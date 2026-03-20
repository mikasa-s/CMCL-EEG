#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FINETUNE_CONFIG="configs/finetune_ds009999.yaml"
CACHE_ROOT="cache/ds009999"
OUTPUT_ROOT="outputs/ds009999"
EPOCHS="0"
BATCH_SIZE="0"
EVAL_BATCH_SIZE="0"
NUM_WORKERS="-1"
PYTHON_EXE=""
FORCE_CPU="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --finetune-config) FINETUNE_CONFIG="$2"; shift 2 ;;
        --cache-root) CACHE_ROOT="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --python-exe) PYTHON_EXE="$2"; shift 2 ;;
        --force-cpu) FORCE_CPU="true"; shift ;;
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

LOSO_ROOT="${CACHE_ROOT}/loso_subjectwise"
if [[ ! -d "${LOSO_ROOT}" ]]; then
    echo "LOSO directory not found: ${LOSO_ROOT}" >&2
    exit 1
fi

FINETUNE_ROOT="${OUTPUT_ROOT}/finetune"
mkdir -p "${FINETUNE_ROOT}"
shopt -s nullglob
fold_dirs=("${LOSO_ROOT}"/fold_*)
if [[ ${#fold_dirs[@]} -eq 0 ]]; then
    echo "No fold_* directories found under ${LOSO_ROOT}" >&2
    exit 1
fi

for fold_dir in "${fold_dirs[@]}"; do
    fold_name="$(basename "${fold_dir}")"
    args=(
        "run_finetune.py"
        "--config" "${FINETUNE_CONFIG}"
        "--train-manifest" "${fold_dir}/manifest_train.csv"
        "--val-manifest" "${fold_dir}/manifest_val.csv"
        "--test-manifest" "${fold_dir}/manifest_test.csv"
        "--root-dir" "${CACHE_ROOT}"
        "--output-dir" "${FINETUNE_ROOT}/${fold_name}"
    )
    if [[ "${EPOCHS}" != "0" ]]; then
        args+=("--epochs" "${EPOCHS}")
    fi
    if [[ "${BATCH_SIZE}" != "0" ]]; then
        args+=("--batch-size" "${BATCH_SIZE}")
    fi
    if [[ "${EVAL_BATCH_SIZE}" != "0" ]]; then
        args+=("--eval-batch-size" "${EVAL_BATCH_SIZE}")
    fi
    if [[ "${NUM_WORKERS}" != "-1" ]]; then
        args+=("--num-workers" "${NUM_WORKERS}")
    fi
    if [[ "${FORCE_CPU}" == "true" ]]; then
        args+=("--force-cpu")
    fi
    echo "[${fold_name}] finetune"
    "${PYTHON}" "${args[@]}"
done

"${PYTHON}" - <<'PY' "${FINETUNE_ROOT}"
import csv
import json
import math
import os
import sys

root = sys.argv[1]
rows = []
for name in sorted(os.listdir(root)):
    if not name.startswith("fold_"):
        continue
    metrics_path = os.path.join(root, name, "test_metrics.json")
    if not os.path.exists(metrics_path):
        continue
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    rows.append({
        "fold": name,
        "accuracy": float(metrics.get("accuracy", float("nan"))),
        "accuracy_std": float(metrics.get("accuracy_std", float("nan"))),
        "macro_f1": float(metrics.get("macro_f1", float("nan"))),
        "macro_f1_std": float(metrics.get("macro_f1_std", float("nan"))),
        "loss": float(metrics.get("loss", float("nan"))),
    })

if not rows:
    raise SystemExit("No fold test_metrics.json files found")

def _mean(values):
    valid = [v for v in values if not math.isnan(v)]
    return float("nan") if not valid else sum(valid) / len(valid)

def _std(values):
    valid = [v for v in values if not math.isnan(v)]
    if len(valid) <= 1:
        return 0.0
    mean = sum(valid) / len(valid)
    return (sum((v - mean) ** 2 for v in valid) / (len(valid) - 1)) ** 0.5

rows.append({
    "fold": "CROSS_FOLD_MEAN_STD",
    "accuracy": _mean([row["accuracy"] for row in rows]),
    "accuracy_std": _std([row["accuracy"] for row in rows]),
    "macro_f1": _mean([row["macro_f1"] for row in rows]),
    "macro_f1_std": _std([row["macro_f1"] for row in rows]),
    "loss": _mean([row["loss"] for row in rows]),
})

out_path = os.path.join(root, "loso_finetune_summary.csv")
with open(out_path, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved summary: {out_path}")
PY
