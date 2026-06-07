"""
BERTweet 专用分类器。

BERTweet 基于 RoBERTa 架构，分类时直接使用最后一层第 0 个 token
的 hidden state，比复用 BERT 的 pooler_output 更贴近 RoBERTa/BERTweet
的常见 fine-tuning 方式。
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from Bertweet.config import BERTWEET_MODEL_NAME, NUM_LABELS


class BertweetRumorClassifier(nn.Module):
    """BERTweet -> first-token hidden state -> Dropout -> Linear。"""

    def __init__(self, model_name: str = BERTWEET_MODEL_NAME, num_labels: int = NUM_LABELS):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_attentions=True)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None and getattr(self.config, "type_vocab_size", 0) > 1:
            model_inputs["token_type_ids"] = token_type_ids

        outputs = self.bert(**model_inputs)
        cls_hidden = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(cls_hidden))

        attentions = None
        if return_attentions and outputs.attentions is not None:
            attentions = outputs.attentions[-1]

        return logits, attentions
