"""
端到端流水线 - 整合训练、评估、单条预测的完整流程

提供统一的命令行入口。
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_device(device_name: str = None) -> str:
    """在导入训练/推理重依赖前完成设备校验。"""
    from src.config import describe_torch_devices, get_device

    try:
        return get_device(device_name)
    except RuntimeError as exc:
        print("\n设备检测失败:")
        print(str(exc))
        print("\nPyTorch 设备状态:")
        print(describe_torch_devices())
        sys.exit(1)


def run_pipeline(args):
    """运行指定的流水线任务"""
    device = resolve_device(args.device)

    if args.stage == "train":
        from src.train import main as train_main
        train_main(device_name=device)

    elif args.stage == "eval":
        from src.inference import evaluate_on_val
        evaluate_on_val(device=device)

    elif args.stage == "predict":
        from src.inference import single_prediction
        if not args.text:
            print("错误: 请使用 --text 参数提供推文文本")
            sys.exit(1)
        single_prediction(args.text, device=device)

    elif args.stage == "all":
        print("=" * 60)
        print("运行完整流水线: 训练 → 评估 → 示例预测")
        print("=" * 60)

        # 1. 训练
        print("\n>>> 阶段一: 模型训练")
        from src.train import main as train_main
        train_main(device_name=device)

        # 2. 评估
        print("\n>>> 阶段二: 验证集评估")
        from src.inference import evaluate_on_val
        evaluate_on_val(device=device)

        # 3. 示例预测
        print("\n>>> 阶段三: 示例预测")
        sample_text = (
            "BREAKING: Police confirm that an armed suspect has been arrested "
            "in connection with the downtown shooting. More details to follow."
        )
        from src.inference import single_prediction
        single_prediction(sample_text, device=device)

    elif args.stage == "tune":
        from src.tune import run_tuning
        run_tuning(
            device_name=device,
            n_trials=args.trials,
            max_epochs=args.tune_epochs,
            metric=args.tune_metric,
            timeout=args.tune_timeout,
            study_name=args.study_name,
            storage=args.storage,
        )

    elif args.stage == "threshold":
        from src.config import BERT_MODEL_PATH, VAL_PATH
        from src.threshold_tune import tune_threshold
        tune_threshold(
            device_name=device,
            model_path=args.model_path or str(BERT_MODEL_PATH),
            data_path=args.data_path or VAL_PATH,
            batch_size=args.batch_size,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        )

    elif args.stage == "tune":
        from src.tune import run_tuning
        run_tuning(
            device_name=device,
            n_trials=args.trials,
            max_epochs=args.tune_epochs,
            metric=args.tune_metric,
            timeout=args.tune_timeout,
            study_name=args.study_name,
            storage=args.storage,
        )

    else:
        print(f"未知阶段: {args.stage}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="RumorDetect - 可解释的谣言检测系统"
    )
    parser.add_argument(
        "--stage", type=str, default="train",
        choices=["train", "eval", "predict", "all", "tune", "threshold"],
        help="运行阶段: train(训练) / eval(评估) / predict(单条预测) / all(全部) / tune(Optuna调参) / threshold(阈值调优)"
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="预测模式下的推文文本"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="运行设备: auto / cpu / cuda / cuda:0 / mps。也可用环境变量 RUMORDETECT_DEVICE 指定。"
    )
    parser.add_argument(
        "--trials", type=int, default=20,
        help="tune 模式下的 Optuna trial 数量。"
    )
    parser.add_argument(
        "--tune-epochs", type=int, default=3,
        help="tune 模式下每个 trial 的最大训练 epoch 数。"
    )
    parser.add_argument(
        "--tune-metric", type=str, default="f1",
        choices=["f1", "accuracy", "precision", "recall"],
        help="tune 模式下优化的验证集指标。"
    )
    parser.add_argument(
        "--tune-timeout", type=int, default=None,
        help="tune 模式下的超时时间，单位秒。"
    )
    parser.add_argument(
        "--study-name", type=str, default="rumordetect-bert-hpo",
        help="tune 模式下的 Optuna study 名称。"
    )
    parser.add_argument(
        "--storage", type=str, default=None,
        help="tune 模式下的 Optuna storage URL，例如 sqlite:///results/optuna.db。"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="threshold 模式下的模型路径，默认使用 config.py 中的 BERT_MODEL_PATH。"
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="threshold 模式下的验证集 CSV 路径，默认使用 config.py 中的 VAL_PATH。"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="threshold 模式下的推理 batch size。"
    )
    parser.add_argument(
        "--threshold-min", type=float, default=0.05,
        help="threshold 模式下扫描的最小阈值。"
    )
    parser.add_argument(
        "--threshold-max", type=float, default=0.95,
        help="threshold 模式下扫描的最大阈值。"
    )
    parser.add_argument(
        "--threshold-step", type=float, default=0.01,
        help="threshold 模式下扫描的阈值步长。"
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
