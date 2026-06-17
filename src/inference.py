"""
推理脚本 - 使用训练好的模型进行预测和解释生成
"""
import os
import sys
import json
import torch
import csv
from typing import List, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    BERT_MODEL_PATH, VAL_PATH, RESULTS_DIR, RETRIEVE_TOP_K, MAX_SEQ_LENGTH,
    get_device, describe_torch_devices
)
from src.bert_classifier import load_model
from src.data_processor import load_csv_data, get_tokenizer, clean_text
from src.dense_retriever import DenseRetriever
from src.llm_explainer import LLMExplainer


class InferenceEngine:
    """
    推理引擎

    整合 BERT 分类 + RAG 检索 + LLM 解释生成 的端到端推理流程。
    """

    def __init__(
        self,
        model_path: str = str(BERT_MODEL_PATH),
        device: str = None,
    ):
        try:
            self.device = torch.device(get_device(device))
        except RuntimeError as exc:
            print("\n设备检测失败:")
            print(str(exc))
            print("\nPyTorch 设备状态:")
            print(describe_torch_devices())
            raise

        print(f"加载模型 (设备: {self.device})...")
        self.model, self.tokenizer = load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        print("加载稠密检索器...")
        self.retriever = DenseRetriever()
        self.retriever._load_index()
        self.retriever._load_corpus_metadata()
        # 重新加载模型（因为 DenseRetriever 初始化时可能会重新下载）
        from sentence_transformers import SentenceTransformer
        from src.config import EMBEDDING_MODEL_NAME
        if self.retriever.model is None:
            self.retriever.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        print("初始化解释生成器...")
        self.explainer = LLMExplainer()

        print("推理引擎初始化完成！")

    def predict_single(self, text: str) -> dict:
        """
        对单条推文进行完整预测

        返回:
            {
                "text": 原始文本,
                "prediction": 0/1,
                "confidence": 置信度,
                "key_evidence": [关键词列表],
                "retrieved_cases": "参考案例文本",
                "explanation": "解释文本"
            }
        """
        # 1. BERT 预测
        cleaned = clean_text(text)
        encoding = self.tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        token_type_ids = encoding.get(
            "token_type_ids", torch.zeros_like(input_ids)
        ).to(self.device)

        with torch.no_grad():
            logits, attentions = self.model(input_ids, attention_mask, token_type_ids, return_attentions=True)
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][prediction].item()

        # 2. 提取关键证据
        key_evidence = []
        if attentions is not None:
            key_evidence = self.model.get_important_tokens(
                input_ids, attentions, self.tokenizer, top_k=10
            )[0]

        # 3. RAG 检索
        try:
            retrieved_cases = self.retriever.retrieve_formatted(text, top_k=RETRIEVE_TOP_K)
        except RuntimeError:
            retrieved_cases = "无相关参考案例。"

        # 4. LLM 解释生成
        explanation = self.explainer.generate(
            text=text,
            prediction=prediction,
            confidence=confidence,
            key_evidence=key_evidence,
            retrieved_cases=retrieved_cases,
        )

        return {
            "text": text,
            "prediction": prediction,
            "prediction_label": "谣言" if prediction == 1 else "非谣言",
            "confidence": confidence,
            "key_evidence": key_evidence,
            "retrieved_cases": retrieved_cases,
            "explanation": explanation,
        }

    def predict_batch(self, texts: List[str]) -> List[dict]:
        """批量预测"""
        results = []
        for text in tqdm(texts, desc="批量推理"):
            result = self.predict_single(text)
            results.append(result)
        return results


def evaluate_on_val(device: str = None):
    """在验证集上运行完整评估"""
    print("=" * 60)
    print("RumorDetect - 验证集评估")
    print("=" * 60)

    # 加载验证集
    ids, texts, labels, events = load_csv_data(VAL_PATH)
    print(f"验证集大小: {len(texts)} 条")

    # 初始化推理引擎
    engine = InferenceEngine(device=device)

    # 批量推理
    results = engine.predict_batch(texts)

    # 计算准确率
    correct = sum(1 for i, r in enumerate(results) if r["prediction"] == labels[i])
    accuracy = correct / len(labels)
    print(f"\n验证集准确率: {accuracy:.4f} ({correct}/{len(labels)})")

    # 保存结果
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = RESULTS_DIR / "val_predictions.json"
    output_data = []
    for i, r in enumerate(results):
        output_data.append({
            "id": ids[i],
            "text": texts[i],
            "true_label": labels[i],
            "predicted_label": r["prediction"],
            "confidence": r["confidence"],
            "key_evidence": r["key_evidence"],
            "event": events[i],
            "explanation": r["explanation"],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"结果已保存至: {output_path}")

    # 同时也保存一个易读的 CSV
    csv_path = RESULTS_DIR / "val_predictions.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "true_label", "predicted_label", "confidence", "event", "explanation"])
        for item in output_data:
            writer.writerow([
                item["id"],
                item["text"],
                item["true_label"],
                item["predicted_label"],
                f"{item['confidence']:.4f}",
                item["event"],
                item["explanation"],
            ])
    print(f"CSV 结果已保存至: {csv_path}")

    return results, accuracy


def single_prediction(text: str, device: str = None):
    """单条推文预测（供交互式使用）"""
    engine = InferenceEngine(device=device)
    result = engine.predict_single(text)

    print("\n" + "=" * 60)
    print("预测结果")
    print("=" * 60)
    print(f"推文: {result['text'][:100]}...")
    print(f"预测: {result['prediction']} ({result['prediction_label']})")
    print(f"置信度: {result['confidence']:.2%}")
    print(f"关键证据: {', '.join(result['key_evidence'])}")
    print(f"\n判断依据:\n{result['explanation']}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RumorDetect 推理")
    parser.add_argument("--mode", type=str, default="eval",
                        choices=["eval", "single"],
                        help="运行模式: eval(评估验证集) / single(单条预测)")
    parser.add_argument("--text", type=str, default=None,
                        help="单条预测时的推文文本")
    parser.add_argument("--device", type=str, default=None,
                        help="运行设备: auto / cpu / cuda / cuda:0 / mps")

    args = parser.parse_args()

    if args.mode == "eval":
        evaluate_on_val(device=args.device)
    elif args.mode == "single":
        if not args.text:
            print("请使用 --text 参数提供推文文本")
        else:
            single_prediction(args.text, device=args.device)
