"""
LLM解释生成器 - 调用交大本地API生成自然语言判断依据

支持 OpenAI 兼容的 API 格式（大多数本地 LLM API 均支持此格式）。
"""
import os
import sys
import json
import requests
from typing import Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    LLM_API_URL, LLM_API_KEY, LLM_MODEL_NAME,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P,
    RETRIEVE_TOP_K
)


# ==================== Prompt 模板 ====================

EXPLANATION_SYSTEM_PROMPT = """你是一个专业的社交媒体谣言分析助手。你的任务是基于给定的推文内容、模型预测结果和参考案例，分析该推文是否属于谣言，并用中文生成清晰、具体的判断依据。

请从以下几个方面进行分析（选择性使用，不要生硬罗列）：
1. **语言风格**：推文是否使用情绪化、夸张或煽动性语言
2. **信息来源**：是否引用可靠来源，或使用模糊的引用方式
3. **事实核查**：推文中的具体陈述是否有事实依据，是否与其他已知信息矛盾
4. **上下文分析**：结合参考案例，该推文与已知的谣言/非谣言模式是否相似
5. **逻辑推理**：推文中的论证是否合理，是否存在逻辑谬误

注意：
- 你的输出应是一段连贯的文字，而不是要点列表
- 指出推文中具体的可疑或可信表述
- 判断依据应基于推文内容和参考案例，而非主观臆断
- 请使用中文回复"""


EXPLANATION_USER_PROMPT_TEMPLATE = """## 推文内容
{text}

## 模型预测结果
预测类别：{prediction_str}（置信度：{confidence:.2%}）

## 模型关注的关键证据
模型在推理过程中重点关注了以下关键词/短语：
{key_evidence}

## 参考案例（训练集中与该推文语义相似的文本及其真实标签）
{retrieved_cases}

请根据以上信息，分析这条推文，用中文生成判断依据。"""


class LLMExplainer:
    """
    LLM 解释生成器

    调用交大本地 API 生成可解释性分析。
    支持 OpenAI 兼容格式，可通过配置适配不同的 API 网关。
    """

    def __init__(
        self,
        api_url: str = LLM_API_URL,
        api_key: str = LLM_API_KEY,
        model_name: str = LLM_MODEL_NAME,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        top_p: float = LLM_TOP_P,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def _check_config(self):
        """检查 API 配置是否有效"""
        if self.api_key == "your-api-key-here" or not self.api_key:
            return False
        if not self.api_url:
            return False
        return True

    def generate(
        self,
        text: str,
        prediction: int,
        confidence: float,
        key_evidence: List[str],
        retrieved_cases: str,
    ) -> str:
        """
        生成判断依据

        参数:
            text: 原始推文文本
            prediction: 预测标签 (0/1)
            confidence: 预测置信度
            key_evidence: BERT关注的关键词列表
            retrieved_cases: RAG检索到的参考案例（已格式化字符串）
        """
        prediction_str = "谣言" if prediction == 1 else "非谣言"
        key_evidence_str = "、".join(key_evidence) if key_evidence else "无突出特征"

        user_prompt = EXPLANATION_USER_PROMPT_TEMPLATE.format(
            text=text,
            prediction_str=prediction_str,
            confidence=confidence,
            key_evidence=key_evidence,
            retrieved_cases=retrieved_cases,
        )

        if not self._check_config():
            # 无有效API配置时返回模拟解释
            return self._generate_fallback_explanation(
                text, prediction_str, confidence, key_evidence_str, retrieved_cases
            )

        return self._call_api(user_prompt)

    def _call_api(self, user_prompt: str) -> str:
        """
        调用交大本地API

        使用 OpenAI 兼容格式。
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()

            # 兼容不同 API 返回格式
            if "choices" in result:
                content = result["choices"][0]["message"]["content"]
            elif "response" in result:
                content = result["response"]
            else:
                content = str(result)

            return content.strip()

        except requests.exceptions.Timeout:
            return self._generate_fallback_explanation_text(
                "API请求超时，请检查网络连接。"
            )
        except requests.exceptions.ConnectionError:
            return self._generate_fallback_explanation_text(
                f"无法连接到API服务器（{self.api_url}），请确认API地址和网络配置。"
            )
        except Exception as e:
            return self._generate_fallback_explanation_text(
                f"API调用出错: {str(e)}"
            )

    def _generate_fallback_explanation(
        self, text, prediction_str, confidence, key_evidence, retrieved_cases
    ):
        """当API不可用时生成基础解释"""
        explanation = (
            f"【模型判断】该推文被判定为「{prediction_str}」（置信度：{confidence:.2%}）。\n\n"
            f"【关键证据】模型重点关注了以下词语：{key_evidence}。\n\n"
        )

        if retrieved_cases and retrieved_cases != "无相关参考案例。":
            explanation += f"【参考案例】{retrieved_cases}\n\n"

        explanation += (
            "【分析说明】以上判断基于BERT深度语义分析模型。当API配置完成后，"
            "本系统将调用大语言模型生成更详细的自然语言判断依据。"
        )
        return explanation

    def _generate_fallback_explanation_text(self, error_msg: str) -> str:
        return (
            f"【解释生成暂不可用】\n{error_msg}\n\n"
            f"请检查 src/config.py 中的 LLM_API_URL 和 LLM_API_KEY 配置是否正确。"
        )


def batch_generate_explanations(
    texts: List[str],
    predictions: List[int],
    confidences: List[float],
    key_evidences: List[List[str]],
    retrieved_cases_list: List[str],
    explainer: Optional[LLMExplainer] = None,
    batch_size: int = 5,
) -> List[str]:
    """
    批量生成解释（带进度条）

    参数:
        texts: 推文文本列表
        predictions: 预测标签列表
        confidences: 置信度列表
        key_evidences: 关键词列表的列表
        retrieved_cases_list: 检索案例字符串列表
        explainer: LLMExplainer 实例，为 None 时自动创建
        batch_size: 并发请求数

    返回:
        解释文本列表
    """
    if explainer is None:
        explainer = LLMExplainer()

    explanations = []
    from tqdm import tqdm

    for i in tqdm(range(len(texts)), desc="生成解释"):
        explanation = explainer.generate(
            text=texts[i],
            prediction=predictions[i],
            confidence=confidences[i],
            key_evidence=key_evidences[i],
            retrieved_cases=retrieved_cases_list[i],
        )
        explanations.append(explanation)

    return explanations
