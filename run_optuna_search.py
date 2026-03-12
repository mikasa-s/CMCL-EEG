from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Optuna automation for EEG-fMRI-Contrastive")
    parser.add_argument("--study-config", type=str, required=True, help="YAML file that defines command, search space, and metric source.")
    parser.add_argument("--mode", type=str, default="", help="Optional study mode, such as full, finetune_only, or pretrain_only.")
    parser.add_argument("--n-trials", type=int, default=None, help="Override study.n_trials from YAML.")
    parser.add_argument("--timeout", type=int, default=None, help="Override study.timeout in seconds.")
    parser.add_argument("--study-name", type=str, default="", help="Override study.name from YAML.")
    parser.add_argument("--output-dir", type=str, default="", help="Override study.output_dir from YAML.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately when a trial command fails.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Study config must be a mapping: {path}")
    return payload


def assign_nested_value(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = payload
    for key in parts[:-1]:
        next_value = cursor.get(key)
        if next_value is None:
            next_value = {}
            cursor[key] = next_value
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot override nested key '{dotted_key}' because '{key}' is not a mapping")
        cursor = next_value
    cursor[parts[-1]] = value


def resolve_path(path_value: str, *, base_dir: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def normalize_study_config(raw: dict[str, Any], args: argparse.Namespace, config_path: Path) -> dict[str, Any]:
    study_cfg = dict(raw.get("study", {}))
    metric_cfg = dict(raw.get("metric", {}))
    parameters_cfg = dict(raw.get("parameters", {}))
    modes_cfg = dict(raw.get("modes", {}))
    runtime_cfg = dict(raw.get("runtime_configs", {}))
    if not study_cfg or not metric_cfg or not parameters_cfg:
        raise ValueError("Study config must contain study, metric, and parameters sections")

    normalized = {
        "config_path": config_path.resolve(),
        "study_name": args.study_name.strip() or str(study_cfg.get("name", config_path.stem)).strip(),
        "direction": str(study_cfg.get("direction", "maximize")).strip().lower(),
        "n_trials": int(args.n_trials if args.n_trials is not None else study_cfg.get("n_trials", 20)),
        "timeout": args.timeout if args.timeout is not None else study_cfg.get("timeout", None),
        "output_dir": resolve_path(args.output_dir.strip() or str(study_cfg.get("output_dir", f"outputs/optuna/{config_path.stem}")), base_dir=PROJECT_ROOT),
        "command": [str(item) for item in study_cfg.get("command", [])],
        "static_args": [str(item) for item in study_cfg.get("static_args", [])],
        "cwd": resolve_path(str(study_cfg.get("cwd", ".")), base_dir=PROJECT_ROOT),
        "output_arg": str(study_cfg.get("output_arg", "")).strip(),
        "metric": {
            "type": str(metric_cfg.get("type", "json")).strip().lower(),
            "path": str(metric_cfg.get("path", "")).strip(),
            "key": str(metric_cfg.get("key", "")).strip(),
            "column": str(metric_cfg.get("column", "")).strip(),
            "row_filter": dict(metric_cfg.get("row_filter", {})),
            "transform": str(metric_cfg.get("transform", "none")).strip().lower(),
        },
        "parameters": parameters_cfg,
        "runtime_configs": {
            "train_base": resolve_path(str(runtime_cfg.get("train_base", "")), base_dir=PROJECT_ROOT) if str(runtime_cfg.get("train_base", "")).strip() else None,
            "finetune_base": resolve_path(str(runtime_cfg.get("finetune_base", "")), base_dir=PROJECT_ROOT) if str(runtime_cfg.get("finetune_base", "")).strip() else None,
            "train_arg": str(runtime_cfg.get("train_arg", "-JointTrainConfig")).strip(),
            "finetune_arg": str(runtime_cfg.get("finetune_arg", "-FinetuneConfig")).strip(),
        },
    }

    mode_name = args.mode.strip() or str(study_cfg.get("default_mode", "")).strip()
    if mode_name:
        if mode_name not in modes_cfg:
            raise ValueError(f"Unknown mode '{mode_name}'. Available: {', '.join(sorted(modes_cfg.keys()))}")
        mode_cfg = dict(modes_cfg[mode_name] or {})
        normalized["study_name"] = str(mode_cfg.get("study_name", normalized["study_name"])).strip()
        if not args.output_dir and mode_cfg.get("output_dir"):
            normalized["output_dir"] = resolve_path(str(mode_cfg["output_dir"]), base_dir=PROJECT_ROOT)
        normalized["static_args"] = normalized["static_args"] + [str(item) for item in mode_cfg.get("static_args", [])]
        if mode_cfg.get("metric"):
            merged_metric = dict(normalized["metric"])
            merged_metric.update(dict(mode_cfg["metric"]))
            normalized["metric"] = merged_metric
        parameter_names = mode_cfg.get("parameter_names")
        if parameter_names:
            normalized["parameters"] = {name: normalized["parameters"][name] for name in parameter_names}

    if not normalized["command"]:
        raise ValueError("study.command must be a non-empty string list")
    if not normalized["output_arg"]:
        raise ValueError("study.output_arg must be configured")
    return normalized


def sample_parameter(trial: Any, name: str, spec: dict[str, Any]) -> Any:
    suggest = str(spec.get("suggest", spec.get("type", ""))).strip().lower()
    if suggest in {"float", "suggest_float"}:
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)), step=spec.get("step"))
    if suggest in {"int", "suggest_int"}:
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)), log=bool(spec.get("log", False)))
    if suggest in {"categorical", "choice", "choices", "suggest_categorical"}:
        choices = spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Parameter '{name}' requires a non-empty choices list")
        return trial.suggest_categorical(name, choices)
    raise ValueError(f"Unsupported parameter suggest type for '{name}': {suggest}")


def apply_config_updates(config_payloads: dict[str, dict[str, Any]], sampled_values: dict[str, Any], parameters_cfg: dict[str, Any]) -> None:
    for name, value in sampled_values.items():
        spec = dict(parameters_cfg[name])
        for update in spec.get("config_updates", []):
            config_name = str(update.get("config", "")).strip()
            dotted_key = str(update.get("key", "")).strip()
            if config_name not in config_payloads or not dotted_key:
                continue
            assign_nested_value(config_payloads[config_name], dotted_key, value)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def extract_metric(metric_cfg: dict[str, Any], output_root: Path) -> float:
    metric_path = output_root / metric_cfg["path"]
    if not metric_path.exists():
        raise FileNotFoundError(f"Metric file not found: {metric_path}")

    if metric_cfg["type"] == "csv":
        with open(metric_path, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        filtered_rows = rows
        for key, expected in metric_cfg.get("row_filter", {}).items():
            filtered_rows = [row for row in filtered_rows if str(row.get(key, "")) == str(expected)]
        if not filtered_rows:
            raise ValueError(f"No metric row matched filter in {metric_path}")
        value = float(filtered_rows[0][metric_cfg["column"]])
    else:
        with open(metric_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        cursor: Any = payload
        for token in metric_cfg["key"].split("."):
            token = token.strip()
            if not token:
                continue
            cursor = cursor[token]
        value = float(cursor)

    if metric_cfg.get("transform") == "negate":
        return -value
    return value


def run_trial_command(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Trial command failed with exit code {completed.returncode}")


def main() -> None:
    args = parse_args()
    study_config_path = resolve_path(args.study_config, base_dir=PROJECT_ROOT)
    study_cfg = normalize_study_config(load_yaml(study_config_path), args, study_config_path)

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is not installed in the current Python environment") from exc

    study_cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    trials_dir = study_cfg["output_dir"] / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(direction=study_cfg["direction"], study_name=study_cfg["study_name"])
    trial_rows: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        sampled_values = {name: sample_parameter(trial, name, spec) for name, spec in study_cfg["parameters"].items()}
        trial_dir = trials_dir / f"trial_{trial.number:04d}"
        trial_output_root = trial_dir / "run_output"
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_output_root.mkdir(parents=True, exist_ok=True)

        config_payloads: dict[str, dict[str, Any]] = {}
        runtime_cfg = study_cfg["runtime_configs"]
        if runtime_cfg.get("train_base") is not None:
            config_payloads["train"] = load_yaml(Path(runtime_cfg["train_base"]))
        if runtime_cfg.get("finetune_base") is not None:
            config_payloads["finetune"] = load_yaml(Path(runtime_cfg["finetune_base"]))
        apply_config_updates(config_payloads, sampled_values, study_cfg["parameters"])

        command = list(study_cfg["command"])
        command.extend(study_cfg["static_args"])
        command.extend([study_cfg["output_arg"], str(trial_output_root)])

        if "train" in config_payloads:
            runtime_train_path = trial_dir / "runtime_train_config.yaml"
            write_yaml(runtime_train_path, config_payloads["train"])
            command.extend([runtime_cfg["train_arg"], str(runtime_train_path)])
        if "finetune" in config_payloads:
            runtime_finetune_path = trial_dir / "runtime_finetune_config.yaml"
            write_yaml(runtime_finetune_path, config_payloads["finetune"])
            command.extend([runtime_cfg["finetune_arg"], str(runtime_finetune_path)])

        for name, value in sampled_values.items():
            spec = dict(study_cfg["parameters"][name])
            if str(spec.get("target", "config")).strip().lower() != "cli":
                continue
            cli_arg = str(spec.get("cli_arg", "")).strip()
            if cli_arg:
                command.extend([cli_arg, str(value)])

        with open(trial_dir / "trial_plan.json", "w", encoding="utf-8") as handle:
            json.dump({"command": command, "sampled_values": sampled_values}, handle, ensure_ascii=False, indent=2)

        run_trial_command(command, cwd=study_cfg["cwd"])
        metric_value = extract_metric(study_cfg["metric"], trial_output_root)
        trial_rows.append({"trial": trial.number, "metric": metric_value, **sampled_values})
        return metric_value

    study.optimize(objective, n_trials=study_cfg["n_trials"], timeout=study_cfg["timeout"], catch=(Exception,) if not args.fail_fast else ())

    if trial_rows:
        with open(study_cfg["output_dir"] / "trials_summary.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trial_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trial_rows)

    completed_trials = [trial for trial in study.trials if trial.value is not None]
    if not completed_trials:
        raise RuntimeError("No Optuna trial completed successfully")

    best_payload = {
        "study_name": study.study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "direction": study.direction.name.lower(),
    }
    with open(study_cfg["output_dir"] / "best_trial.json", "w", encoding="utf-8") as handle:
        json.dump(best_payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(best_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Optuna run failed: {exc}", file=sys.stderr)
        raise