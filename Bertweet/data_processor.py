"""
BERTweet 实验数据处理。
"""
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from Bertweet.config import (
    BATCH_SIZE, BERTWEET_MODEL_NAME, EVAL_BATCH_SIZE, MAX_SEQ_LENGTH,
    RANDOM_SEED, TOKENIZER_KWARGS, TRAIN_PATH, TRAIN_VAL_SPLIT
)
from src.data_processor import load_csv_data


def clean_text(text: str) -> str:
    """
    按 BERTweet 预训练习惯规范化推文。

    BERTweet 使用 HTTPURL 和 @USER 作为 URL/用户占位符。
    """
    text = re.sub(r"http\S+|www\S+|https\S+", "HTTPURL", text, flags=re.IGNORECASE)
    text = re.sub(r"@\w+", "@USER", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_tokenizer(model_name: str = BERTWEET_MODEL_NAME):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, **TOKENIZER_KWARGS)


class BertweetRumorDataset(Dataset):
    """BERTweet 谣言检测数据集。"""

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

        token_type_ids = encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"]))

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_type_ids": token_type_ids.flatten(),
            "label": torch.tensor(label, dtype=torch.long),
            "text": self.texts[idx],
        }


def get_data_loaders(
    train_path: Path = TRAIN_PATH,
    val_split: float = TRAIN_VAL_SPLIT,
    batch_size: int = BATCH_SIZE,
    eval_batch_size: int = EVAL_BATCH_SIZE,
    max_len: int = MAX_SEQ_LENGTH,
) -> Tuple[DataLoader, DataLoader, Dict]:
    tokenizer = get_tokenizer()

    train_ids, train_texts, train_labels, train_events = load_csv_data(train_path)

    split_idx = int(len(train_texts) * val_split)
    random.seed(RANDOM_SEED)
    indices = list(range(len(train_texts)))
    random.shuffle(indices)

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_texts_split = [train_texts[i] for i in train_idx]
    train_labels_split = [train_labels[i] for i in train_idx]
    val_texts_split = [train_texts[i] for i in val_idx]
    val_labels_split = [train_labels[i] for i in val_idx]

    extra_info = {
        "all_train_texts": train_texts,
        "all_train_labels": train_labels,
        "all_train_events": train_events,
        "all_train_ids": train_ids,
        "val_texts": val_texts_split,
        "val_labels": val_labels_split,
        "val_indices": val_idx,
    }

    train_dataset = BertweetRumorDataset(train_texts_split, train_labels_split, tokenizer, max_len)
    val_dataset = BertweetRumorDataset(val_texts_split, val_labels_split, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)

    return train_loader, val_loader, extra_info
