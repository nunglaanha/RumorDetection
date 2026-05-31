"""
端到端流水线 - 整合训练、评估、单条预测的完整流程

提供统一的命令行入口。
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_pipeline(args):
    """运行指定的流水线任务"""
    if args.stage == "train":
        from src.train import main as train_main
        train_main()

    elif args.stage == "eval":
        from src.inference import evaluate_on_val
        evaluate_on_val()

    elif args.stage == "predict":
        from src.inference import single_prediction
        if not args.text:
            print("错误: 请使用 --text 参数提供推文文本")
            sys.exit(1)
        single_prediction(args.text)

    elif args.stage == "all":
        print("=" * 60)
        print("运行完整流水线: 训练 → 评估 → 示例预测")
        print("=" * 60)

        # 1. 训练
        print("\n>>> 阶段一: 模型训练")
        from src.train import main as train_main
        train_main()

        # 2. 评估
        print("\n>>> 阶段二: 验证集评估")
        from src.inference import evaluate_on_val
        evaluate_on_val()

        # 3. 示例预测
        print("\n>>> 阶段三: 示例预测")
        sample_text = (
            "BREAKING: Police confirm that an armed suspect has been arrested "
            "in connection with the downtown shooting. More details to follow."
        )
        from src.inference import single_prediction
        single_prediction(sample_text)

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
        choices=["train", "eval", "predict", "all", "tune"],
        help="运行阶段: train(训练) / eval(评估) / predict(单条预测) / all(全部) / tune(Optuna调参)"
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

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
