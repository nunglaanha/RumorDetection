"""
BERT分类器 - 基于BERT的谣言检测模型，支持注意力权重提取
"""
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoConfig
from typing import Tuple, List, Optional

from src.config import BERT_MODEL_NAME, NUM_LABELS, MAX_SEQ_LENGTH


class BertRumorClassifier(nn.Module):
    """
    BERT谣言分类器

    结构: BERT → [CLS]池化 → Dropout → 线性分类头
    支持前向传播时返回注意力权重，用于可解释性分析
    """

    def __init__(self, model_name: str = BERT_MODEL_NAME, num_labels: int = NUM_LABELS):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_attentions=True)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        参数:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            token_type_ids: token 类型 IDs
            return_attentions: 是否返回注意力权重

        返回:
            logits: 分类 logits
            attentions: 注意力权重（可选）
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
        )

        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        if return_attentions and outputs.attentions is not None:
            # outputs.attentions 是 (layers, batch, heads, seq, seq) 的元组
            # 取最后一层的注意力权重
            return logits, outputs.attentions[-1]
        return logits, None

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预测并返回标签、概率和注意力

        返回:
            predictions: 预测类别 (0/1)
            probabilities: 预测概率分布
            attention_weights: [batch, heads, seq_len, seq_len]
        """
        self.eval()
        with torch.no_grad():
            logits, attentions = self.forward(
                input_ids, attention_mask, token_type_ids, return_attentions=True
            )
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

        return predictions, probabilities, attentions

    def get_important_tokens(
        self,
        input_ids: torch.Tensor,
        attention_weights: torch.Tensor,
        tokenizer,
        top_k: int = 10,
    ) -> List[List[str]]:
        """
        从注意力权重中提取重要 token

        方法: 对 [CLS] token 的注意力分数在各头上取平均，
        选择 attention_mask 范围内分数最高的 top_k 个 token

        参数:
            input_ids: [batch, seq_len]
            attention_weights: [batch, heads, seq_len, seq_len]
            tokenizer: 用于将 id 解码为文本
            top_k: 返回的重要 token 数量

        返回:
            每个样本的重要 token 列表
        """
        # [CLS] 对所有 token 的注意力: [batch, heads, seq_len]
        cls_attention = attention_weights[:, :, 0, :]  # [CLS] → 所有位置

        # 跨注意力头取平均: [batch, seq_len]
        avg_attention = cls_attention.mean(dim=1)

        # 创建 attention_mask（排除 padding 和特殊 token）
        batch_size, seq_len = input_ids.shape
        results = []

        for b in range(batch_size):
            scores = avg_attention[b]
            ids = input_ids[b]

            # 排除 [PAD] (0), [CLS] (101), [SEP] (102) token
            valid_mask = ~torch.isin(ids, torch.tensor([0, 101, 102], device=ids.device))
            valid_scores = scores.masked_fill(~valid_mask, -float("inf"))

            # 取 top_k
            top_indices = torch.topk(valid_scores, min(top_k, (valid_mask).sum().item())).indices
            top_tokens = [tokenizer.decode(ids[i].item()) for i in top_indices]
            results.append(top_tokens)

        return results


def save_model(model: BertRumorClassifier, tokenizer, save_path: str):
    """保存模型和分词器"""
    import os
    os.makedirs(save_path, exist_ok=True)
    model.bert.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    # 保存分类头
    torch.save(model.classifier.state_dict(), os.path.join(save_path, "classifier_head.pt"))


def load_model(model_path: str, num_labels: int = NUM_LABELS) -> BertRumorClassifier:
    """从保存路径加载模型"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = BertRumorClassifier(model_name=model_path, num_labels=num_labels)
    # 加载分类头
    import os
    classifier_path = os.path.join(model_path, "classifier_head.pt")
    if os.path.exists(classifier_path):
        model.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
    return model, tokenizer
