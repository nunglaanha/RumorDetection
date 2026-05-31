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

    else:
        print(f"未知阶段: {args.stage}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="RumorDetect - 可解释的谣言检测系统"
    )
    parser.add_argument(
        "--stage", type=str, default="train",
        choices=["train", "eval", "predict", "all"],
        help="运行阶段: train(训练) / eval(评估) / predict(单条预测) / all(全部)"
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="预测模式下的推文文本"
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
