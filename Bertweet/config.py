"""
BERTweet 实验配置。

该目录用于独立试验 Twitter 领域预训练模型，不影响 src/ 下的 BERT baseline。
"""
from src.config import (
    DATA_DIR, PROJECT_ROOT, TRAIN_PATH, VAL_PATH,
    RANDOM_SEED, NUM_LABELS, get_device, describe_torch_devices
)


EXPERIMENT_DIR = PROJECT_ROOT / "Bertweet"
MODEL_DIR = EXPERIMENT_DIR / "models"
RESULTS_DIR = EXPERIMENT_DIR / "results"
BERTWEET_MODEL_PATH = MODEL_DIR / "bertweet_rumor_classifier"

BERTWEET_MODEL_NAME = "vinai/bertweet-base"
TOKENIZER_KWARGS = {"normalization": True}

MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 3
TRAIN_VAL_SPLIT = 0.8
