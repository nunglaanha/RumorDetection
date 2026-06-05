"""
配置文件 - 集中管理所有超参数和路径设置
"""
import os
from pathlib import Path


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==================== 路径配置 ====================
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_PATH = DATA_DIR / "train_cleaned.csv"
VAL_PATH = DATA_DIR / "val.csv"

# ================ 直接下载的预训练模型保存路径 =============
# 包括原始BERT模型 bert-base-uncased 和 用于RAG的嵌入模型 all-MiniLM-L6-v2
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, "models", "pretrained")

# ==================== 微调后模型保存路径 ====================
BERT_MODEL_PATH = MODEL_DIR / "bert_rumor_classifier"
BERT_TOKENIZER_PATH = MODEL_DIR / "bert_tokenizer"
FAISS_INDEX_PATH = MODEL_DIR / "dense_index.faiss"
EMBEDDINGS_CACHE_PATH = MODEL_DIR / "train_embeddings.npy"
EMBEDDING_MODEL_PATH = MODEL_DIR / "sentence_transformer"

# ==================== BERT 训练参数 ====================
BERT_MODEL_NAME = "bert-base-uncased"       # 预训练模型名称
MAX_SEQ_LENGTH = 128                         # 最大序列长度
BATCH_SIZE = 16                              # 训练批次大小
EVAL_BATCH_SIZE = 32                         # 评估批次大小
EPOCHS = 10                                  # 最大训练轮数
LEARNING_RATE = 2e-5                         # 学习率
WEIGHT_DECAY = 0.01                          # 权重衰减
WARMUP_RATIO = 0.1                           # 预热比例
EARLY_STOPPING_PATIENCE = 3                  # 早停耐心值
TRAIN_VAL_SPLIT = 0.8                        # 训练集划分比例
NUM_LABELS = 2                               # 分类类别数

# ==================== 密集检索(RAG)参数 ====================
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"    # sentence-transformers 嵌入模型
RETRIEVE_TOP_K = 5                           # 检索返回的最相似样本数

# ==================== LLM API 配置 ====================
# 交大本地 API 配置
LLM_API_URL = os.getenv(
    "LLM_API_URL",
    "https://models.sjtu.edu.cn/api/v1"
)
LLM_API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3.5-27b")

# LLM 生成参数
LLM_TEMPERATURE = 0.3        # 生成温度（越低越确定）
LLM_MAX_TOKENS = 512         # 最大生成长度
LLM_TOP_P = 0.9              # top-p 采样

# ==================== 通用配置 ====================
RANDOM_SEED = 42
def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = get_device()