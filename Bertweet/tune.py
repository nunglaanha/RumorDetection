"""
BERTweet 实验 Optuna 调参入口。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="RumorDetect - BERTweet Optuna 自动调参")
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
    parser.add_argument("--study-name", type=str, default="rumordetect-bertweet-hpo",
                        help="Optuna study 名称")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL，例如 sqlite:///Bertweet/results/optuna.db")
    args = parser.parse_args()

    from Bertweet.config import BERTWEET_MODEL_NAME, EVAL_BATCH_SIZE, MAX_SEQ_LENGTH, RESULTS_DIR
    from Bertweet.classifier import BertweetRumorClassifier
    from Bertweet.data_processor import get_data_loaders
    from src.tune import run_tuning

    run_tuning(
        device_name=args.device,
        n_trials=args.trials,
        max_epochs=args.tune_epochs,
        metric=args.tune_metric,
        timeout=args.tune_timeout,
        study_name=args.study_name,
        storage=args.storage,
        output_dir=RESULTS_DIR,
        model_name=BERTWEET_MODEL_NAME,
        data_loader_fn=get_data_loaders,
        model_class=BertweetRumorClassifier,
        max_seq_length=MAX_SEQ_LENGTH,
        eval_batch_size=EVAL_BATCH_SIZE,
        config_target="Bertweet/config.py",
        train_command="python -m Bertweet.train --device cuda",
    )


if __name__ == "__main__":
    main()
