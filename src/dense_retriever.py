"""
稠密检索器 - 基于sentence-transformers + FAISS的语义检索模块
用于RAG（检索增强生成）中的相似案例检索
"""
import os
import sys
import numpy as np
import torch
from typing import List, Tuple, Optional

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_PATH,
    FAISS_INDEX_PATH, EMBEDDINGS_CACHE_PATH,
    RETRIEVE_TOP_K, MODEL_DIR
)


class DenseRetriever:
    # 使用 sentence-transformers 编码文本为稠密向量，通过 FAISS 进行高效的相似度检索，为 LLM 提供相关参考案例。
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        index_path: str = str(FAISS_INDEX_PATH),
        embeddings_cache_path: str = str(EMBEDDINGS_CACHE_PATH),
    ):
        self.model_name = model_name
        self.index_path = index_path
        self.embeddings_cache_path = embeddings_cache_path
        self.model = None
        self.index = None
        self.corpus_texts: List[str] = []
        self.corpus_labels: List[int] = []
        self._load_or_initialize()

    def _load_or_initialize(self):
        from sentence_transformers import SentenceTransformer

        # 优先从本地路径加载
        local_path = str(EMBEDDING_MODEL_PATH)
        if os.path.isdir(local_path) and os.listdir(local_path):
            print(f"从本地加载嵌入模型: {local_path}")
            self.model = SentenceTransformer(local_path)
        else:
            # 尝试从 HuggingFace 加载（失败时回退到国内镜像）
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                import httpx
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                print(f"直连 HuggingFace 失败，尝试镜像站下载 {self.model_name} ...")
                self.model = SentenceTransformer(self.model_name)

        if os.path.exists(self.index_path) and os.path.exists(self.embeddings_cache_path):
            self._load_index()
            self._load_corpus_metadata()

    def _load_index(self):
        import faiss
        self.index = faiss.read_index(self.index_path)

    def _load_corpus_metadata(self):
        metadata_path = self.embeddings_cache_path.replace(".npy", "_metadata.npz")
        if os.path.exists(metadata_path):
            data = np.load(metadata_path, allow_pickle=True)
            self.corpus_texts = data["texts"].tolist()
            self.corpus_labels = data["labels"].tolist()

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            self._load_or_initialize()
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)

    def build_index(self, texts: List[str], labels: List[int]):
        import faiss

        print(f"编码第 {len(texts)} 条文本...")
        embeddings = self.encode(texts)

        # 构建 FAISS 索引 (L2 距离)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        # 保存索引
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)

        self.corpus_texts = texts
        self.corpus_labels = labels
        metadata_path = self.embeddings_cache_path.replace(".npy", "_metadata.npz")
        np.savez(
            metadata_path,
            texts=np.array(texts, dtype=object),
            labels=np.array(labels, dtype=int),
        )

        np.save(self.embeddings_cache_path, embeddings)

        print(f"索引构建完成: {self.index.ntotal} 条向量")

    def retrieve(
        self, query: str, top_k: int = RETRIEVE_TOP_K
    ) -> List[Tuple[str, int, float]]:
        # 检索与查询最相似的 top_k 条文本
        if self.index is None:
            raise RuntimeError("索引未构建，请先调用 build_index()")

        query_vec = self.encode([query])
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.corpus_texts):
                results.append((
                    self.corpus_texts[idx],
                    self.corpus_labels[idx],
                    float(distances[0][i]),
                ))

        return results

    def retrieve_formatted(
        self, query: str, top_k: int = RETRIEVE_TOP_K
    ) -> str:
        # 检索最相似的字符串，格式化之后返回
        results = self.retrieve(query, top_k)

        if not results:
            return "无相关参考案例。"

        formatted = ""
        for i, (text, label, dist) in enumerate(results, 1):
            label_text = "谣言" if label == 1 else "非谣言"
            formatted += f"案例{i}: \"{text[:200]}\"\n"
            formatted += f"  真实标签: {label_text}\n\n"

        return formatted.strip()


def build_faiss_index(texts: List[str], labels: List[int]):
    # 构建并保存FAISS索引
    retriever = DenseRetriever()
    retriever.build_index(texts, labels)
    return retriever
