"""
BERTweet 实验训练入口。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(device_name: str = None):
    import torch
    from torch.optim import AdamW
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from Bertweet.config import (
        BATCH_SIZE, BERTWEET_MODEL_NAME, BERTWEET_MODEL_PATH, EARLY_STOPPING_PATIENCE,
        EPOCHS, LEARNING_RATE, RESULTS_DIR, TOKENIZER_KWARGS, WEIGHT_DECAY,
        WARMUP_RATIO, describe_torch_devices, get_device
    )
    from Bertweet.data_processor import get_data_loaders
    from Bertweet.classifier import BertweetRumorClassifier
    from src.bert_classifier import save_model
    from src.train import evaluate, set_seed, train_epoch

    print("=" * 60)
    print("RumorDetect - BERTweet 谣言检测模型训练")
    print("=" * 60)

    set_seed()
    try:
        device = torch.device(get_device(device_name))
    except RuntimeError as exc:
        print("\n设备检测失败:")
        print(str(exc))
        print("\nPyTorch 设备状态:")
        print(describe_torch_devices())
        raise

    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"CUDA 设备: {torch.cuda.get_device_name(device)}")

    print("\n[1/4] 加载 BERTweet 数据...")
    train_loader, val_loader, _ = get_data_loaders()
    print(f"  训练集: {len(train_loader.dataset)} 条")
    print(f"  验证集: {len(val_loader.dataset)} 条")
    print(f"  Batch size: {BATCH_SIZE}")

    print("\n[2/4] 初始化 BERTweet 模型...")
    model = BertweetRumorClassifier(model_name=BERTWEET_MODEL_NAME)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(BERTWEET_MODEL_NAME, **TOKENIZER_KWARGS)

    print("\n[3/4] 开始训练...")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_f1 = 0.0
    best_val_metrics = None
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch}/{EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_metrics = evaluate(model, val_loader, device)

        print(f"  训练 Loss: {train_loss:.4f}")
        print(f"  验证 Loss: {val_metrics['loss']:.4f}")
        print(f"  验证 Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  验证 F1: {val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_val_metrics = {k: float(v) for k, v in val_metrics.items()}
            patience_counter = 0
            best_model_state = {
                "bert": model.bert.state_dict(),
                "classifier": model.classifier.state_dict(),
            }
            print(f"  新最佳模型 (F1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  早停计数: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("  触发早停！")
                break

    if best_model_state is None:
        raise RuntimeError("训练没有产生可保存的最佳模型。")

    print("\n[4/4] 保存 BERTweet 最佳模型...")
    model.bert.load_state_dict(best_model_state["bert"])
    model.classifier.load_state_dict(best_model_state["classifier"])
    os.makedirs(BERTWEET_MODEL_PATH, exist_ok=True)
    save_model(model, tokenizer, str(BERTWEET_MODEL_PATH))
    print(f"  模型已保存至: {BERTWEET_MODEL_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / "train_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": BERTWEET_MODEL_NAME,
                "best_val_f1": best_val_f1,
                "best_val_metrics": best_val_metrics,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "warmup_ratio": WARMUP_RATIO,
                "epochs": EPOCHS,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  训练指标已保存至: {metrics_path}")

    print("\n" + "=" * 60)
    print("BERTweet 训练完成！")
    print(f"最佳验证 F1: {best_val_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RumorDetect - BERTweet 训练")
    parser.add_argument("--device", type=str, default=None,
                        help="运行设备: auto / cpu / cuda / cuda:0 / mps")
    args = parser.parse_args()
    main(device_name=args.device)
