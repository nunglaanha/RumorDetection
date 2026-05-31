"""
数据处理器 - 负责数据加载、清洗、分词和数据集构建
"""
import re
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from src.config import (
    TRAIN_PATH, VAL_PATH, BERT_MODEL_NAME,
    MAX_SEQ_LENGTH, BATCH_SIZE, EVAL_BATCH_SIZE, RANDOM_SEED
)


def clean_text(text: str) -> str:
    """
    清洗推文文本：
    - 去除 URL
    - 去除 @用户名（保留#话题标签作为语义特征）
    - 合并多余空白
    """
    text = re.sub(r"http\S+|www\S+|https\S+", "[URL]", text, flags=re.IGNORECASE)
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class RumorDataset(Dataset):
    """谣言检测数据集"""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = clean_text(str(self.texts[idx]))
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros(self.max_len)).flatten(),
            "label": torch.tensor(label, dtype=torch.long),
            "text": self.texts[idx],      # 保留原始文本用于解释生成
        }


def load_csv_data(file_path: Path) -> Tuple[List[str], List[int], List[str], List[str]]:
    """
    加载CSV数据
    返回: (ids, texts, labels, events)
    """
    ids, texts, labels, events = [], [], [], []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["id"])
            texts.append(row["text"])
            labels.append(int(row["label"]))
            events.append(row["event"])
    return ids, texts, labels, events


def get_data_loaders(
    train_path: Path = TRAIN_PATH,
    val_path: Path = VAL_PATH,
    val_split: float = 0.8,
    batch_size: int = BATCH_SIZE,
    eval_batch_size: int = EVAL_BATCH_SIZE,
    max_len: int = MAX_SEQ_LENGTH,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    加载数据并创建 DataLoader

    返回:
        train_loader, val_loader, extra_info
        extra_info 包含原始文本列表等后续用到的信息
    """
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    # 加载训练集
    train_ids, train_texts, train_labels, train_events = load_csv_data(train_path)

    # 划分训练/验证
    split_idx = int(len(train_texts) * val_split)
    import random
    random.seed(RANDOM_SEED)
    indices = list(range(len(train_texts)))
    random.shuffle(indices)

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_texts_split = [train_texts[i] for i in train_idx]
    train_labels_split = [train_labels[i] for i in train_idx]
    val_texts_split = [train_texts[i] for i in val_idx]
    val_labels_split = [train_labels[i] for i in val_idx]

    # 额外信息
    extra_info = {
        "all_train_texts": train_texts,
        "all_train_labels": train_labels,
        "all_train_events": train_events,
        "all_train_ids": train_ids,
        "val_texts": val_texts_split,
        "val_labels": val_labels_split,
        "val_indices": val_idx,
    }

    # 构建 Dataset 和 DataLoader
    train_dataset = RumorDataset(train_texts_split, train_labels_split, tokenizer, max_len)
    val_dataset = RumorDataset(val_texts_split, val_labels_split, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)

    return train_loader, val_loader, extra_info


def get_tokenizer(model_name: str = BERT_MODEL_NAME):
    """获取分词器"""
    return AutoTokenizer.from_pretrained(model_name)
