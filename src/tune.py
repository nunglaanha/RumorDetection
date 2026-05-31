"""
Optuna 超参数搜索 - 自动寻找 BERT 微调参数
"""
import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    BERT_MODEL_NAME, EVAL_BATCH_SIZE, MAX_SEQ_LENGTH, RANDOM_SEED,
    RESULTS_DIR, describe_torch_devices, get_device
)


BATCH_SIZE_CHOICES = [16, 32, 64, 128]
LEARNING_RATE_RANGE = (1e-5, 5e-5)
WEIGHT_DECAY_RANGE = (0.0, 0.05)
WARMUP_RATIO_RANGE = (0.0, 0.2)


def _is_oom_error(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save_study_results(study, output_dir: Path, metric: str, max_epochs: int) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    best = {
        "metric": metric,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": study.best_params,
        "max_epochs_per_trial": max_epochs,
        "n_trials": len(study.trials),
    }

    best_path = output_dir / "optuna_best_params.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    trials_path = output_dir / "optuna_trials.csv"
    param_names = sorted({name for trial in study.trials for name in trial.params})
    user_attr_names = sorted({name for trial in study.trials for name in trial.user_attrs})
    fieldnames = ["number", "state", "value"] + param_names + user_attr_names
    with open(trials_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trial in study.trials:
            row = {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
            }
            row.update(trial.params)
            row.update(trial.user_attrs)
            writer.writerow(row)

    print(f"\n最佳参数已保存: {best_path}")
    print(f"全部 trial 记录已保存: {trials_path}")
    return best


def _print_best_config(best: Dict):
    params = best["best_params"]
    print("\n建议写入 src/config.py 的参数:")
    print(f"BATCH_SIZE = {params['batch_size']}")
    print(f"LEARNING_RATE = {params['learning_rate']:.8g}")
    print(f"WEIGHT_DECAY = {params['weight_decay']:.8g}")
    print(f"WARMUP_RATIO = {params['warmup_ratio']:.8g}")
    print("\n然后完整训练:")
    print("python -m src.pipeline --stage train --device cuda")


def run_tuning(
    device_name: Optional[str] = None,
    n_trials: int = 20,
    max_epochs: int = 3,
    metric: str = "f1",
    timeout: Optional[int] = None,
    study_name: str = "rumordetect-bert-hpo",
    storage: Optional[str] = None,
    output_dir: Path = RESULTS_DIR,
):
    """
    运行 Optuna 超参数搜索。

    每个 trial 只训练少量 epoch，用验证集指标评估参数好坏。
    """
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "未安装 optuna。请先运行: pip install optuna"
        ) from exc
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    from src.bert_classifier import BertRumorClassifier
    from src.data_processor import get_data_loaders
    from src.train import evaluate, set_seed, train_epoch

    if metric not in {"f1", "accuracy", "precision", "recall"}:
        raise ValueError("metric 只能是 f1 / accuracy / precision / recall")

    try:
        device = torch.device(get_device(device_name))
    except RuntimeError as exc:
        print("\n设备检测失败:")
        print(str(exc))
        print("\nPyTorch 设备状态:")
        print(describe_torch_devices())
        raise

    print("=" * 60)
    print("RumorDetect - Optuna 超参数搜索")
    print("=" * 60)
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"CUDA 设备: {torch.cuda.get_device_name(device)}")
    print(f"目标指标: val_{metric}")
    print(f"trial 数量: {n_trials}")
    print(f"每个 trial 最大 epoch: {max_epochs}")

    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES)
        learning_rate = trial.suggest_float(
            "learning_rate", LEARNING_RATE_RANGE[0], LEARNING_RATE_RANGE[1], log=True
        )
        weight_decay = trial.suggest_float(
            "weight_decay", WEIGHT_DECAY_RANGE[0], WEIGHT_DECAY_RANGE[1]
        )
        warmup_ratio = trial.suggest_float(
            "warmup_ratio", WARMUP_RATIO_RANGE[0], WARMUP_RATIO_RANGE[1]
        )

        print("\n" + "-" * 60)
        print(
            f"Trial {trial.number}: batch={batch_size}, "
            f"lr={learning_rate:.2e}, wd={weight_decay:.4f}, warmup={warmup_ratio:.3f}"
        )

        set_seed(RANDOM_SEED)
        best_metric = 0.0
        model = optimizer = scheduler = train_loader = val_loader = None

        try:
            train_loader, val_loader, _ = get_data_loaders(
                batch_size=batch_size,
                eval_batch_size=max(EVAL_BATCH_SIZE, batch_size),
                max_len=MAX_SEQ_LENGTH,
            )
            model = BertRumorClassifier(model_name=BERT_MODEL_NAME)
            model.to(device)

            optimizer = AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            total_steps = len(train_loader) * max_epochs
            warmup_steps = int(total_steps * warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

            for epoch in range(1, max_epochs + 1):
                train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
                val_metrics = evaluate(model, val_loader, device)
                current = val_metrics[metric]
                best_metric = max(best_metric, current)

                trial.report(current, step=epoch)
                trial.set_user_attr(f"epoch_{epoch}_train_loss", train_loss)
                for name, value in val_metrics.items():
                    trial.set_user_attr(f"epoch_{epoch}_val_{name}", float(value))

                print(
                    f"  Epoch {epoch}/{max_epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_{metric}={current:.4f} | "
                    f"val_f1={val_metrics['f1']:.4f} | "
                    f"val_acc={val_metrics['accuracy']:.4f}"
                )

                if trial.should_prune():
                    raise optuna.TrialPruned()

            return best_metric

        except RuntimeError as exc:
            if _is_oom_error(exc):
                print("  CUDA 显存不足，剪枝该 trial。")
                raise optuna.TrialPruned() from exc
            raise
        finally:
            del model, optimizer, scheduler, train_loader, val_loader
            _cleanup_cuda()

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=bool(storage),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    complete_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete_trials:
        raise RuntimeError("没有完成的 trial。请降低 batch size 搜索范围或检查训练环境。")

    best = _save_study_results(study, Path(output_dir), metric, max_epochs)
    print("\n" + "=" * 60)
    print(f"最佳 trial: {best['best_trial']}")
    print(f"最佳 val_{metric}: {best['best_value']:.4f}")
    print(f"最佳参数: {best['best_params']}")
    print("=" * 60)
    _print_best_config(best)
    return best


def main():
    parser = argparse.ArgumentParser(description="RumorDetect - Optuna 自动调参")
    parser.add_argument("--device", type=str, default=None,
                        help="运行设备: auto / cpu / cuda / cuda:0 / mps")
    parser.add_argument("--trials", type=int, default=20,
                        help="Optuna trial 数量")
    parser.add_argument("--tune-epochs", type=int, default=3,
                        help="每个 trial 的最大训练 epoch 数")
    parser.add_argument("--tune-metric", type=str, default="f1",
                        choices=["f1", "accuracy", "precision", "recall"],
                        help="调参优化目标")
    parser.add_argument("--tune-timeout", type=int, default=None,
                        help="调参超时时间，单位秒")
    parser.add_argument("--study-name", type=str, default="rumordetect-bert-hpo",
                        help="Optuna study 名称")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL，例如 sqlite:///results/optuna.db")
    args = parser.parse_args()

    run_tuning(
        device_name=args.device,
        n_trials=args.trials,
        max_epochs=args.tune_epochs,
        metric=args.tune_metric,
        timeout=args.tune_timeout,
        study_name=args.study_name,
        storage=args.storage,
    )


if __name__ == "__main__":
    main()
