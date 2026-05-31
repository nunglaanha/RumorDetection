"""
阈值调优 - 在验证集上搜索最佳谣言概率阈值。
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    BERT_MODEL_PATH, EVAL_BATCH_SIZE, MAX_SEQ_LENGTH, RESULTS_DIR, VAL_PATH,
    describe_torch_devices, get_device
)


def _encode_batch(tokenizer, texts: List[str], device: torch.device):
    from src.data_processor import clean_text

    cleaned = [clean_text(text) for text in texts]
    encoding = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding.get("token_type_ids", torch.zeros_like(input_ids)).to(device)
    return input_ids, attention_mask, token_type_ids


def _metrics(labels, probs, threshold: float) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    preds = (probs >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="binary", zero_division=0)),
        "precision": float(precision_score(labels, preds, average="binary", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="binary", zero_division=0)),
        "positive_predictions": int(preds.sum()),
    }


def tune_threshold(
    device_name: str = None,
    model_path: str = str(BERT_MODEL_PATH),
    data_path: Path = VAL_PATH,
    output_dir: Path = RESULTS_DIR,
    batch_size: int = EVAL_BATCH_SIZE,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_step: float = 0.01,
) -> dict:
    """在验证集上搜索使 F1 最大的 P(谣言) 阈值。"""
    try:
        device = torch.device(get_device(device_name))
    except RuntimeError as exc:
        print("\n设备检测失败:")
        print(str(exc))
        print("\nPyTorch 设备状态:")
        print(describe_torch_devices())
        raise

    print("=" * 60)
    print("RumorDetect - 阈值调优")
    print("=" * 60)
    print(f"使用设备: {device}")
    print(f"模型路径: {model_path}")
    print(f"验证数据: {data_path}")

    from src.bert_classifier import load_model
    from src.data_processor import load_csv_data

    ids, texts, labels, events = load_csv_data(Path(data_path))
    labels_np = np.array(labels, dtype=int)
    print(f"验证集大小: {len(texts)} 条")
    print(f"真实谣言数: {int(labels_np.sum())}")

    print("\n加载分类模型...")
    model, tokenizer = load_model(model_path)
    model.to(device)
    model.eval()

    rumor_probs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="计算 P(谣言)"):
            batch_texts = texts[start:start + batch_size]
            input_ids, attention_mask, token_type_ids = _encode_batch(tokenizer, batch_texts, device)
            logits, _ = model(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            rumor_probs.extend(probs.detach().cpu().numpy().tolist())

    probs_np = np.array(rumor_probs, dtype=float)
    thresholds = np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step)
    curve = [_metrics(labels_np, probs_np, float(threshold)) for threshold in thresholds]
    best = max(curve, key=lambda row: (row["f1"], row["accuracy"], row["threshold"]))
    default = _metrics(labels_np, probs_np, 0.5)

    print("\n默认阈值 0.50:")
    print(
        f"  F1={default['f1']:.4f}, Acc={default['accuracy']:.4f}, "
        f"P={default['precision']:.4f}, R={default['recall']:.4f}"
    )
    print("最佳阈值:")
    print(
        f"  threshold={best['threshold']:.2f}, F1={best['f1']:.4f}, "
        f"Acc={best['accuracy']:.4f}, P={best['precision']:.4f}, R={best['recall']:.4f}"
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_path": str(model_path),
        "data_path": str(data_path),
        "num_examples": len(texts),
        "default_threshold": default,
        "best_threshold": best,
    }

    summary_path = output_dir / "threshold_tuning.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    curve_path = output_dir / "threshold_curve.csv"
    with open(curve_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)

    predictions_path = output_dir / "val_probabilities.csv"
    best_threshold = best["threshold"]
    with open(predictions_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "text", "true_label", "event", "rumor_probability",
            "prediction_at_0.50", "prediction_at_best_threshold"
        ])
        for sample_id, text, label, event, prob in zip(ids, texts, labels, events, probs_np):
            writer.writerow([
                sample_id,
                text,
                label,
                event,
                f"{prob:.6f}",
                int(prob >= 0.5),
                int(prob >= best_threshold),
            ])

    print(f"\n阈值摘要已保存至: {summary_path}")
    print(f"阈值曲线已保存至: {curve_path}")
    print(f"样本概率已保存至: {predictions_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="RumorDetect - 阈值调优")
    parser.add_argument("--device", type=str, default=None,
                        help="运行设备: auto / cpu / cuda / cuda:0 / mps")
    parser.add_argument("--model-path", type=str, default=str(BERT_MODEL_PATH),
                        help="模型路径")
    parser.add_argument("--data-path", type=str, default=str(VAL_PATH),
                        help="验证集 CSV 路径")
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE,
                        help="推理 batch size")
    parser.add_argument("--threshold-min", type=float, default=0.05,
                        help="最小阈值")
    parser.add_argument("--threshold-max", type=float, default=0.95,
                        help="最大阈值")
    parser.add_argument("--threshold-step", type=float, default=0.01,
                        help="阈值步长")
    args = parser.parse_args()

    tune_threshold(
        device_name=args.device,
        model_path=args.model_path,
        data_path=Path(args.data_path),
        batch_size=args.batch_size,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
    )


if __name__ == "__main__":
    main()
