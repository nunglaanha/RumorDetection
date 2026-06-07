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
LEARNING_RATE = 4.763628595029452e-05        # 学习率
WEIGHT_DECAY = 0.041622                     # 权重衰减
WARMUP_RATIO = 0.042468                      # 预热比例
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
    "https://models.sjtu.edu.cn/api/v1/chat/completions"
)
LLM_API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3.5-27b")

# LLM 生成参数
LLM_TEMPERATURE = 0.3        # 生成温度（越低越确定）
LLM_MAX_TOKENS = 512         # 最大生成长度
LLM_TOP_P = 0.9              # top-p 采样

# ==================== 通用配置 ====================
RANDOM_SEED = 42


def get_device(requested: str = None) -> str:
    """
    获取训练/推理设备。

    requested 可传入 "auto"、"cpu"、"cuda"、"cuda:0"、"mps"。
    也可通过环境变量 RUMORDETECT_DEVICE 指定。
    """
    import torch

    requested = requested or os.getenv("RUMORDETECT_DEVICE", "auto")
    requested = requested.lower()

    if requested in ("auto", ""):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cpu":
        return "cpu"

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "请求使用 CUDA，但当前 PyTorch 检测不到 CUDA。请检查是否安装了 CUDA 版 torch、"
                "是否在分配到 GPU 的节点/容器中运行，以及 CUDA_VISIBLE_DEVICES 是否屏蔽了 GPU。"
            )
        return requested

    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("请求使用 MPS，但当前 PyTorch 检测不到可用的 Apple GPU/MPS。")
        return "mps"

    raise ValueError(f"不支持的设备参数: {requested}")


def describe_torch_devices() -> str:
    """返回 PyTorch 看到的设备状态，便于定位服务器环境问题。"""
    import torch

    lines = [
        f"torch: {torch.__version__}",
        f"torch.version.cuda: {torch.version.cuda}",
        f"cuda.is_available: {torch.cuda.is_available()}",
        f"cuda.device_count: {torch.cuda.device_count()}",
        f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')}",
    ]
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            lines.append(f"cuda:{idx}: {torch.cuda.get_device_name(idx)}")
    return "\n".join(lines)


# 兼容旧代码的默认值；运行入口会在执行时重新调用 get_device() 读取参数/环境变量。
DEVICE = get_device("auto")
