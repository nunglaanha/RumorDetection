"""
训练脚本 - 微调BERT分类器并构建稠密检索索引
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    BERT_MODEL_NAME, BERT_MODEL_PATH, BERT_TOKENIZER_PATH,
    FAISS_INDEX_PATH, RESULTS_DIR, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    WARMUP_RATIO, EARLY_STOPPING_PATIENCE, RANDOM_SEED, MAX_SEQ_LENGTH,
    get_device, describe_torch_devices
)
from src.data_processor import get_data_loaders, load_csv_data
from src.bert_classifier import BertRumorClassifier, save_model
from src.dense_retriever import build_faiss_index


def set_seed(seed: int = RANDOM_SEED):
    """设置随机种子以确保可复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: BertRumorClassifier,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
) -> float:
    """训练一个epoch，返回平均loss"""
    model.train()
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits, _ = model(input_ids, attention_mask, token_type_ids)
        loss = loss_fn(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def evaluate(
    model: BertRumorClassifier,
    dataloader: torch.utils.data.DataLoader,
    device: str,
) -> dict:
    """评估模型，返回各项指标"""
    model.eval()
    all_preds, all_labels = [], []
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="binary"),
        "precision": precision_score(all_labels, all_preds, average="binary", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="binary", zero_division=0),
    }
    return metrics


def main(device_name: str = None):
    """主训练流程"""
    print("=" * 60)
    print("RumorDetect - BERT谣言检测模型训练")
    print("=" * 60)

    # 设置
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

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    train_loader, val_loader, extra_info = get_data_loaders()
    print(f"  训练集: {len(train_loader.dataset)} 条")
    print(f"  验证集: {len(val_loader.dataset)} 条")

    # 2. 初始化模型
    print("\n[2/5] 初始化BERT模型...")
    model = BertRumorClassifier(model_name=BERT_MODEL_NAME)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    # 3. 设置优化器和学习率调度器
    print("\n[3/5] 设置优化器...")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 4. 训练
    print("\n[4/5] 开始训练...")
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch}/{EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_metrics = evaluate(model, val_loader, device)

        print(f"  训练 Loss: {train_loss:.4f}")
        print(f"  验证 Loss: {val_metrics['loss']:.4f}")
        print(f"  验证 Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  验证 F1: {val_metrics['f1']:.4f}")

        # 记录 loss 和指标
        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_metrics["loss"], 6))
        history["val_accuracy"].append(round(val_metrics["accuracy"], 6))
        history["val_f1"].append(round(val_metrics["f1"], 6))

        # 早停和模型保存
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            best_model_state = {
                "bert": model.bert.state_dict(),
                "classifier": model.classifier.state_dict(),
            }
            print(f"  ✓ 新最佳模型 (F1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  早停计数: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("  触发早停！")
                break

    # 加载最佳模型
    print("\n[5/5] 保存最佳模型...")
    model.bert.load_state_dict(best_model_state["bert"])
    model.classifier.load_state_dict(best_model_state["classifier"])
    os.makedirs(BERT_MODEL_PATH, exist_ok=True)
    save_model(model, tokenizer, str(BERT_MODEL_PATH))
    print(f"  模型已保存至: {BERT_MODEL_PATH}")

    # 保存训练历史（loss 曲线数据）
    import json
    history_path = RESULTS_DIR / "training_history.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"  训练历史已保存至: {history_path}")

    # 绘制 loss 曲线
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color="tab:blue")
        ax1.plot(history["epoch"], history["train_loss"], marker="o", color="tab:blue",
                 label="Train Loss")
        ax1.plot(history["epoch"], history["val_loss"], marker="s", color="tab:orange",
                 label="Val Loss")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel("F1", color="tab:green")
        ax2.plot(history["epoch"], history["val_f1"], marker="^", color="tab:green",
                 label="Val F1", linestyle="--")
        ax2.tick_params(axis="y", labelcolor="tab:green")
        ax2.legend(loc="upper right")

        plt.title("Training Loss & Validation F1 Curve")
        plt.tight_layout()
        curve_path = RESULTS_DIR / "loss_curve.png"
        plt.savefig(curve_path, dpi=150)
        plt.close()
        print(f"  Loss 曲线已保存至: {curve_path}")
    except ImportError:
        print("  matplotlib 未安装，跳过 loss 曲线绘制")

    # 构建FAISS索引（RAG检索用）
    print("\n构建稠密检索索引...")
    all_texts = extra_info["all_train_texts"]
    all_labels_list = extra_info["all_train_labels"]
    build_faiss_index(all_texts, all_labels_list)
    print(f"  索引已保存至: {FAISS_INDEX_PATH}")

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"最佳验证 F1: {best_val_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
