#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch


def round_to_multiple(value: float, multiple: int) -> int:
    value = int(round(value))
    return max(multiple, (value // multiple) * multiple)


def suggest_optuna_params(trial) -> Dict[str, object]:
    """
    Model search space used by ScreenSBERT.from_optuna_trial().
    Kept here so the entire optuna policy is centralized in one place.
    """
    # Fixed values to reduce search space.
    fixed_width = 128
    fixed_height = 256
    fixed_num_buckets = 32
    fixed_max_distance = 128
    fixed_log_base = 2.0
    fixed_layer_scale_init = 0.1
    fixed_proj_init_scale = 0.1

    embed_dim = trial.suggest_categorical("embed_dim", [256, 384, 512, 640, 768])
    num_heads = trial.suggest_categorical("num_heads", [4, 8, 16])
    if embed_dim % num_heads != 0:
        valid_heads = [h for h in (16, 12, 8, 6, 4, 3, 2, 1) if embed_dim % h == 0]
        num_heads = valid_heads[0]

    d_ff_mult = trial.suggest_categorical("d_ff_mult", [2.0, 3.0])
    d_ff = round_to_multiple(embed_dim * d_ff_mult, 64)

    # Keep projection bottlenecks deterministic to reduce search dimensions.
    proj_hidden_dim = max(64, round_to_multiple(embed_dim / 4, 32))
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1, 0.15, 0.2])
    num_layers = trial.suggest_categorical("num_layers", [2, 3, 4, 6, 8])

    return {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "d_ff": d_ff,
        "dropout": dropout,
        "attn_dropout": dropout,
        "width": fixed_width,
        "height": fixed_height,
        "num_buckets": fixed_num_buckets,
        "max_distance": fixed_max_distance,
        "log_base": fixed_log_base,
        "layer_scale_init": fixed_layer_scale_init,
        "proj_dropout": dropout,
        "proj_init_scale": fixed_proj_init_scale,
        "function_proj_hidden_dim": proj_hidden_dim,
        "vision_proj_hidden_dim": proj_hidden_dim,
        "text_proj_hidden_dim": proj_hidden_dim,
    }


def _cli_flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("", flush=True)
    print("[RUN]", " ".join(cmd), flush=True)
    print("", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def resolve_model_config_from_best_params(best_params: Dict[str, object]) -> Dict[str, object]:
    fixed_width = 128
    fixed_height = 256
    fixed_num_buckets = 32
    fixed_max_distance = 128
    fixed_log_base = 2.0
    fixed_layer_scale_init = 0.1
    fixed_proj_init_scale = 0.1

    embed_dim = int(best_params.get("embed_dim", 768))
    num_heads = int(best_params.get("num_heads", 8))
    if embed_dim % num_heads != 0:
        valid_heads = [h for h in (16, 12, 8, 6, 4, 3, 2, 1) if embed_dim % h == 0]
        num_heads = valid_heads[0]

    d_ff_mult = float(best_params.get("d_ff_mult", 2.0))
    d_ff = round_to_multiple(embed_dim * d_ff_mult, 64)
    dropout = float(best_params.get("dropout", 0.1))
    num_layers = int(best_params.get("num_layers", 4))
    proj_hidden_dim = max(64, round_to_multiple(embed_dim / 4, 32))

    return {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "d_ff": d_ff,
        "dropout": dropout,
        "attn_dropout": dropout,
        "width": fixed_width,
        "height": fixed_height,
        "num_buckets": fixed_num_buckets,
        "max_distance": fixed_max_distance,
        "log_base": fixed_log_base,
        "layer_scale_init": fixed_layer_scale_init,
        "proj_dropout": dropout,
        "proj_init_scale": fixed_proj_init_scale,
        "function_proj_hidden_dim": proj_hidden_dim,
        "vision_proj_hidden_dim": proj_hidden_dim,
        "text_proj_hidden_dim": proj_hidden_dim,
    }


def parse_args() -> argparse.Namespace:
    this_dir = Path(__file__).resolve().parent
    project_root = this_dir.parent

    parser = argparse.ArgumentParser(
        description="Run Optuna search first, then final Screen-SBERT training with best params."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "dataset" / "gui_parsing",
        help="Parsed feature dataset root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=this_dir / "runs" / "optuna_pipeline",
        help="Root directory to store optuna and final training outputs",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Run name (default: timestamp)",
    )
    parser.add_argument("--train-apps", type=str, default="")
    parser.add_argument("--ood-apps", type=str, default="")
    parser.add_argument("--train-classes-per-app", type=int, default=16)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda / cuda:0 / cpu",
    )
    parser.add_argument("--episodes-per-app", type=int, default=0)
    parser.add_argument(
        "--log-interval-steps",
        type=int,
        default=60,
        help="Print running train metrics every N steps in both optuna and final train",
    )
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--contrastive-margin", type=float, default=0.1)
    parser.add_argument("--contrastive-margin-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    parser.add_argument("--optuna-epochs", type=int, default=30)
    parser.add_argument("--final-epochs", type=int, default=30)
    parser.add_argument("--checkpoint-name", type=str, default="final_best.pt")

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.07)
    final_opt_group = parser.add_mutually_exclusive_group()
    final_opt_group.add_argument(
        "--final-use-best-optimizer",
        dest="final_use_best_optimizer",
        action="store_true",
        help="Use best trial's lr/weight_decay/temperature when present (default: on)",
    )
    final_opt_group.add_argument(
        "--no-final-use-best-optimizer",
        dest="final_use_best_optimizer",
        action="store_false",
        help="Do not override final optimizer params with best-trial values",
    )
    parser.set_defaults(final_use_best_optimizer=True)

    final_train_hparam_group = parser.add_mutually_exclusive_group()
    final_train_hparam_group.add_argument(
        "--final-use-best-training-hparams",
        dest="final_use_best_training_hparams",
        action="store_true",
        help="Use best trial's contrastive_margin/contrastive_margin_weight/grad_clip_norm when present (default: on)",
    )
    final_train_hparam_group.add_argument(
        "--no-final-use-best-training-hparams",
        dest="final_use_best_training_hparams",
        action="store_false",
        help="Do not override final training hyperparams with best-trial values",
    )
    parser.set_defaults(final_use_best_training_hparams=True)

    parser.add_argument(
        "--n-trials",
        type=int,
        default=200,
        help="Target number of counted trials (delegated to train_contrastive)",
    )
    parser.add_argument("--study-name", type=str, default="screen_sbert_search")
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--optuna-timeout-sec", type=int, default=0)
    parser.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop training early if val_bal_acc does not improve for this many consecutive epochs (<=0 disables)",
    )
    tune_optimizer_group = parser.add_mutually_exclusive_group()
    tune_optimizer_group.add_argument(
        "--optuna-tune-optimizer",
        dest="optuna_tune_optimizer",
        action="store_true",
        help="Tune lr/weight_decay/temperature during optuna search (default: on)",
    )
    tune_optimizer_group.add_argument(
        "--no-optuna-tune-optimizer",
        dest="optuna_tune_optimizer",
        action="store_false",
        help="Disable optimizer hyperparameter tuning during optuna search",
    )
    parser.set_defaults(optuna_tune_optimizer=True)

    tune_training_group = parser.add_mutually_exclusive_group()
    tune_training_group.add_argument(
        "--optuna-tune-training-hparams",
        dest="optuna_tune_training_hparams",
        action="store_true",
        help="Tune contrastive_margin/contrastive_margin_weight/grad_clip_norm during optuna search (default: on)",
    )
    tune_training_group.add_argument(
        "--no-optuna-tune-training-hparams",
        dest="optuna_tune_training_hparams",
        action="store_false",
        help="Disable training hyperparameter tuning during optuna search",
    )
    parser.set_defaults(optuna_tune_training_hparams=True)

    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable used to run train_contrastive.py",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    this_dir = Path(__file__).resolve().parent
    project_root = this_dir.parent
    train_script = this_dir / "train_contrastive.py"
    if not train_script.is_file():
        raise FileNotFoundError(f"train_contrastive.py not found: {train_script}")

    run_name = args.run_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root.resolve() / run_name
    optuna_dir = run_root / "optuna"
    final_dir = run_root / "final_train"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PIPELINE] run_name={run_name}", flush=True)
    print(f"[PIPELINE] dataset_root={args.dataset_root.resolve()}", flush=True)
    print(f"[PIPELINE] output_run_root={run_root}", flush=True)
    print(f"[PIPELINE] device={args.device}", flush=True)

    common_base_args = [
        "--dataset-root",
        str(args.dataset_root.resolve()),
        "--train-classes-per-app",
        str(args.train_classes_per_app),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--episodes-per-app",
        str(args.episodes_per_app),
        "--log-interval-steps",
        str(args.log_interval_steps),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
    ]
    if args.train_apps.strip():
        common_base_args.extend(["--train-apps", args.train_apps.strip()])
    if args.ood_apps.strip():
        common_base_args.extend(["--ood-apps", args.ood_apps.strip()])

    optuna_cmd = [
        args.python_bin,
        str(train_script),
        "--mode",
        "optuna",
        "--save-dir",
        str(optuna_dir),
        "--epochs",
        str(args.optuna_epochs),
        "--n-trials",
        str(args.n_trials),
        "--study-name",
        f"{args.study_name}_{run_name}",
        "--sampler",
        args.sampler,
        "--optuna-timeout-sec",
        str(args.optuna_timeout_sec),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--temperature",
        str(args.temperature),
        "--contrastive-margin",
        str(args.contrastive_margin),
        "--contrastive-margin-weight",
        str(args.contrastive_margin_weight),
        "--grad-clip-norm",
        str(args.grad_clip_norm),
        *common_base_args,
    ]
    if args.storage.strip():
        optuna_cmd.extend(["--storage", args.storage.strip()])
    if args.optuna_tune_optimizer:
        optuna_cmd.append("--optuna-tune-optimizer")
    else:
        optuna_cmd.append("--no-optuna-tune-optimizer")
    if args.optuna_tune_training_hparams:
        optuna_cmd.append("--optuna-tune-training-hparams")
    else:
        optuna_cmd.append("--no-optuna-tune-training-hparams")

    run_cmd(optuna_cmd, cwd=project_root)

    best_trial_path = optuna_dir / "optuna_best_trial.json"
    if not best_trial_path.is_file():
        raise FileNotFoundError(f"Optuna best trial file not found: {best_trial_path}")
    best_trial_payload = json.loads(best_trial_path.read_text(encoding="utf-8"))
    best_params = dict(best_trial_payload.get("best_params", {}))
    model_config = resolve_model_config_from_best_params(best_params)

    final_lr = args.lr
    final_weight_decay = args.weight_decay
    final_temperature = args.temperature
    if args.final_use_best_optimizer:
        if "lr" in best_params:
            final_lr = float(best_params["lr"])
        if "weight_decay" in best_params:
            final_weight_decay = float(best_params["weight_decay"])
        if "temperature" in best_params:
            final_temperature = float(best_params["temperature"])

    final_contrastive_margin = args.contrastive_margin
    final_contrastive_margin_weight = args.contrastive_margin_weight
    final_grad_clip_norm = args.grad_clip_norm
    if args.final_use_best_training_hparams:
        if "contrastive_margin" in best_params:
            final_contrastive_margin = float(best_params["contrastive_margin"])
        if "contrastive_margin_weight" in best_params:
            final_contrastive_margin_weight = float(best_params["contrastive_margin_weight"])
        if "grad_clip_norm" in best_params:
            final_grad_clip_norm = float(best_params["grad_clip_norm"])

    train_cmd = [
        args.python_bin,
        str(train_script),
        "--mode",
        "train",
        "--save-dir",
        str(final_dir),
        "--checkpoint-name",
        args.checkpoint_name,
        "--epochs",
        str(args.final_epochs),
        "--lr",
        str(final_lr),
        "--weight-decay",
        str(final_weight_decay),
        "--temperature",
        str(final_temperature),
        "--contrastive-margin",
        str(final_contrastive_margin),
        "--contrastive-margin-weight",
        str(final_contrastive_margin_weight),
        "--grad-clip-norm",
        str(final_grad_clip_norm),
        *common_base_args,
    ]
    for key, value in model_config.items():
        train_cmd.extend([_cli_flag(key), str(value)])

    run_cmd(train_cmd, cwd=project_root)

    summary = {
        "run_name": run_name,
        "dataset_root": str(args.dataset_root.resolve()),
        "output_run_root": str(run_root),
        "optuna_dir": str(optuna_dir),
        "final_train_dir": str(final_dir),
        "best_trial_file": str(best_trial_path),
        "best_params": best_params,
        "resolved_model_config": model_config,
        "final_optimizer": {
            "lr": final_lr,
            "weight_decay": final_weight_decay,
            "temperature": final_temperature,
        },
        "final_training_hparams": {
            "contrastive_margin": final_contrastive_margin,
            "contrastive_margin_weight": final_contrastive_margin_weight,
            "grad_clip_norm": final_grad_clip_norm,
        },
        "early_stopping": {
            "patience": args.early_stopping_patience,
        },
    }
    summary_path = run_root / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("", flush=True)
    print("[PIPELINE] completed successfully", flush=True)
    print(f"[PIPELINE] summary: {summary_path}", flush=True)
    print(f"[PIPELINE] final checkpoint: {final_dir / args.checkpoint_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
