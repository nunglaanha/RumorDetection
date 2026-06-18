# RumorDetect: 可解释的谣言检测系统

基于 **BERT 微调 + RAG + 大语言模型解释生成** 的三阶段复合架构，实现对社交媒体推文的谣言检测与可解释性分析。

## 项目概述

| 阶段 | 技术 | 功能 |
|------|---------|------|
| **阶段一** | BERT (`bert-large-uncased`) | 推文二分类（谣言/非谣言），提取注意力权重 |
| **阶段二** | RAG检索：sentence-transformers + FAISS | 稠密语义检索，召回训练集中相似案例 |
| **阶段三** | 交大本地 LLM API | 基于预测结果+检索案例，生成自然语言判断依据 |

### 输入/输出

- **输入**：一条英文推文
- **输出1**：预测的二分类标签（0 非谣言/1 谣言）以及置信度
- **输出2**：中文自然语言判断依据解释，包括
  - 由分类模型注意力提取的关键词
  - LLM得到的判断依据和理由

---

## 环境要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **操作系统** | Windows / Linux / macOS | Linux (Ubuntu 20.04+) |
| **Python** | 3.8+ | 3.10+ |
| **内存** | 8 GB | 16 GB |
| **GPU** | 可选（4GB+ VRAM） | NVIDIA RTX 3060 / 4060 及以上 |
| **CUDA** | 可选 | CUDA 11.8+ |

> **注意**：如果没有 GPU，BERT 训练和推理可以在 CPU 上运行，仅速度较慢（训练约 1 小时，推理每条约 50ms）

---

## 快速开始

### 1. 克隆或创建项目

```bash
cd RumorDetect
```

### 2. 安装环境

```bash
conda create -n nis4307 python=3.10 -y
conda activate nis4307
```

#### 2.1 安装 PyTorch（根据 CUDA 版本选择）

先运行 `nvidia-smi` 查看 CUDA Version，然后根据下表选择对应命令：

| nvidia-smi 显示 CUDA | pip install 命令 |
|---|---|
| CUDA 12.6 | `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124` |
| CUDA 12.4 / 12.5 | `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124` |
| CUDA 12.2 / 12.3 | `pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121` |
| CUDA 12.1 | `pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121` |
| CUDA 11.8 | `pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118` |
| 无 GPU / 驱动过旧 | `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu` |

> **说明**：CUDA 12.2/12.3 无专用 PyTorch 源，但 cu121 编译的二进制可向前兼容至 CUDA 12.3 driver。
`.../whl/cu124` 源提供 torch 2.4.0 ~ 2.6.x，`.../whl/cu121` 源提供 torch 2.1.0 ~ 2.3.x。

#### 2.2 安装其余依赖

PyTorch 安装完成后，执行以下命令安装剩余依赖（pip 会自动跳过已安装的 torch）：

```bash
pip install -r requirements.txt
```

若无法连接PyPi源，请设置镜像源如下：
```bash
pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

#### 2.3 HuggingFace 镜像设置（可选）

若无法访问 HuggingFace Hub，可在安装前设置镜像源：
```bash
# Linux / macOS
export HF_ENDPOINT=https://hf-mirror.com
# Windows CMD
set HF_ENDPOINT=https://hf-mirror.com
# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"
```
设置后 `transformers` 和 `sentence-transformers` 将自动从 `hf-mirror.com` 下载预训练模型。

### 3. 配置数据

确保 `data/` 目录下包含 `train.csv` 和 `val.csv`，文件格式如下：

| id | text | label | event |
|----|------|-------|-------|
| 536824260615237632 | Swiss museum confirms... | 1 | 0 |

#### 数据清洗（已完成，无需操作）

原始训练集存在标签重复和冲突（同一文本被标为不同的谣言/非谣言标签），在训练前通过`data/data_conflict_detect.py`检查。

根据发现的冲突结果，人工判断每条冲突的正确标签，通过`data/data_clean.py`生成清洗后的文件`train_cleaned.csv`.

清洗后将 `src/config.py` 中的 `TRAIN_PATH`设置为指向 `train_cleaned.csv`，训练时使用清洗后的数据

### 4. 配置交大API

#### 环境变量方式（推荐）

```bash
# Linux / macOS
export LLM_API_URL="https://models.sjtu.edu.cn/api/v1/chat/completions"
export LLM_API_KEY="your-actual-api-key"
export LLM_MODEL_NAME="qwen3.5-27b"

# Windows CMD
set LLM_API_URL=https://models.sjtu.edu.cn/api/v1/chat/completions
set LLM_API_KEY=your-actual-api-key
set LLM_MODEL_NAME=qwen3.5-27b

# Windows PowerShell
$env:LLM_API_URL="https://models.sjtu.edu.cn/api/v1/chat/completions"
$env:LLM_API_KEY="your-actual-api-key"
$env:LLM_MODEL_NAME=qwen3.5-27b
```

### 配置文件方式

直接编辑 [src/config.py](src/config.py) 中的以下变量：

```python
LLM_API_URL = "https://models.sjtu.edu.cn/api/v1/chat/completions"
LLM_API_KEY = "your-actual-api-key"
LLM_MODEL_NAME = "qwen3.5-27b"
```
如果未配置 API，系统会使用内置的模板化解释作为降级方案，不影响分类功能。

## 5. 手动下载预训练模型（自动下载失败备用）

训练和推理需要加载两个预训练模型。代码会在首次运行时自动检查本地路径，若不存在则自动从 HuggingFace Hub 下载。如果服务器无法访问 HuggingFace，可先通过下方的方式手动下载。

### 需要下载的模型

| 模型 | 用途 | 大小 | 本地存放路径 |
|------|------|-------|-------------|
| `bert-large-uncased` | BERT 谣言分类器 | 约1.3G | `models/pretrained/bert-large-uncased/` |
| `all-MiniLM-L6-v2` | RAG 语义嵌入模型 | 约80 MB | `models/sentence_transformer/` |

### 方式一：通过 HuggingFace 下载（推荐）
将以下 python 代码保存为 `download_models_hf.py` 放在项目根目录，然后执行：

```bash
cd RumorDetect

# 手动设置环境变量
# Linux / macOS
export HF_ENDPOINT=https://hf-mirror.com
# Windows CMD
set HF_ENDPOINT=https://hf-mirror.com
# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"

# 运行安装脚本
python download_models_hf.py
```

```python 
# download_models_hf.py
from pathlib import Path
from huggingface_hub import snapshot_download

project_root = Path(__file__).resolve().parent

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ---------- 1. 下载 BERT ----------
print("正在下载 bert-large-uncased ...")
snapshot_download(
    "bert-large-uncased",
    local_dir=str(project_root / "models" / "pretrained" / "bert-large-uncased"),
    ignore_patterns=["*.h5", "*.ot", "*.onnx", "*.msgpack", "flax_model.*"],
)
print("[OK] bert-large-uncased 已下载到 models/pretrained/bert-large-uncased/")

# ---------- 2. 下载 Sentence Transformer ----------
from sentence_transformers import SentenceTransformer

print("正在下载 all-MiniLM-L6-v2 ...")
model = SentenceTransformer("all-MiniLM-L6-v2")
save_path = str(project_root / "models" / "sentence_transformer")
model.save(save_path)
print(f"[OK] all-MiniLM-L6-v2 已保存到 {save_path}/")
```


### 方式二：通过 ModelScope 下载（HuggingFace 不可用时备选）

```bash
pip install modelscope
```

将以下python代码保存为 `download_models_ms.py` 放在项目根目录，然后执行：

```bash
cd RumorDetect
python download_models_ms.py
```

```python
# download_models_ms.py
from pathlib import Path
import os

project_root = Path(__file__).resolve().parent

# 跳过不必要的文件格式
IGNORE_PATTERNS = ["*.onnx", "*.ot", "*.msgpack", "*.h5", "flax_model.*"]

# ---------- 1. 下载 BERT ----------
from modelscope import snapshot_download

print("从 ModelScope 下载 bert-large-uncased ...")
snapshot_download(
    "google-bert/bert-large-uncased",
    local_dir=str(project_root / "models" / "pretrained" / "bert-large-uncased"),
    ignore_file_pattern=IGNORE_PATTERNS,
)
print("[OK] bert-large-uncased 已下载到 models/pretrained/bert-large-uncased/")

# ---------- 2. 下载 Sentence Transformer ----------
from sentence_transformers import SentenceTransformer

print("从 ModelScope 下载 all-MiniLM-L6-v2 ...")
model_dir = snapshot_download(
    "sentence-transformers/all-MiniLM-L6-v2",
    ignore_file_pattern=IGNORE_PATTERNS + ["model.onnx"],
)
model = SentenceTransformer(model_dir)
save_path = str(project_root / "models" / "sentence_transformer")
model.save(save_path)
print(f"[OK] all-MiniLM-L6-v2 已保存到 {save_path}/")
```

> **说明**：`ignore_file_pattern` 跳过 ONNX、Flax、TensorFlow 等非 PyTorch 格式文件，只下载 PyTorch 所需的权重和配置文件。这对下载速度有明显提升，且不影响代码正常运行。

### 检查 models 文件放置路径

将下载好的 `models/` 目录整体上传到项目根目录即可，目录结构应与下列一致：

```
RumorDetect/
└── models/
    ├── pretrained/
    │   └── bert-large-uncased/          # BERT 预训练模型（扁平目录）
    │       ├── config.json
    │       ├── model.safetensors        # 或 pytorch_model.bin
    │       ├── tokenizer.json
    │       └── ...
    └── sentence_transformer/            # all-MiniLM-L6-v2
        ├── config.json
        ├── modules.json
        ├── sentence_bert_config.json
        ├── 0_Transformer/
        ├── 1_Pooling/
        └── 2_Normalize/
```

> **注意**：如果手动放置了模型文件，训练时代码检测到 `models/pretrained/bert-large-uncased/config.json` 已存在，就不会再触发联网下载。

---

## 训练模型

```bash
cd RumorDetect
python -m src.pipeline --stage train
```

训练过程将：

1. **加载数据**：从 `data/train.csv` 读取 2840 条推文
2. **划分数据集**：80% 训练集 / 20% 验证集
3. **微调 BERT**：使用交叉熵损失，AdamW 优化器，对BERT模型进行微调；若验证集 F1 连续 3 轮不提升即停止
4. **保存模型**：将最佳模型保存至 `models/bert_rumor_classifier/`
5. **构建检索索引**：对全部训练集构建 FAISS 稠密索引，保存至 `models/dense_index.faiss`，用于后续RAG检索

### 训练输出
示例输出如下：
```
============================================================
RumorDetect - BERT谣言检测模型训练
============================================================
使用设备: cuda

[1/5] 加载数据...
  训练集: 2272 条
  验证集: 568 条

[2/5] 初始化BERT模型...
[3/5] 设置优化器...
[4/5] 开始训练...
========================================
Epoch 1/10
  训练 Loss: 0.5214
  验证 Loss: 0.4123
  验证 Accuracy: 0.8239
  验证 F1: 0.8145
  ✓ 新最佳模型 (F1=0.8145)
...
========================================
Epoch 4/10
  验证 F1: 0.8412
  ✓ 新最佳模型 (F1=0.8412)

[5/5] 保存最佳模型...
构建稠密检索索引...
编码 2840 条文本...
索引构建完成: 2840 条向量
============================================================
训练完成！
最佳验证 F1: 0.8412
============================================================
```

---

## 评测模型

在验证集上运行完整评估（分类 + 解释生成）：

```bash
python -m src.pipeline --stage eval
```

输出结果保存在：

- `results/val_predictions.json`：包含每条推文的完整预测信息
- `results/val_predictions.csv`：易读的表格格式

### 评测输出示例

```
============================================================
RumorDetect - 验证集评估
============================================================
验证集大小: 401 条
加载模型 (设备: cuda)...
加载稠密检索器...
初始化解释生成器...

推理引擎初始化完成！
批量推理: 100%|██████████████████████████████████████████████████████████████| 401/401 [00:18<00:00, 21.99it/s]

验证集准确率: 0.8803 (353/401)
结果已保存至: RumorDetection/results/val_predictions.json
CSV 结果已保存至: RumorDetection/results/val_predictions.csv
```

---

## 单条预测

对自定义推文进行检测和解释：

```bash
python -m src.pipeline --stage predict --text "BREAKING: Government officials have confirmed that a new policy will be announced tomorrow."
```

### 预测输出示例

```
============================================================
预测结果
============================================================
推文: BREAKING: Government officials have confirmed that a new policy will be announced tomorrow....
预测: 非谣言
置信度: 92.35%
关键证据: confirmed, officials, BREAKING, government, announced, policy

判断依据:
该推文使用了官方来源的表述（"Government officials have confirmed"），
语言风格客观中立，未使用情绪化或煽动性词汇。虽然没有具体细节，
但表述方式符合正规新闻发布模式，与训练集中非谣言案例的表述风格一致。
因此判定为非谣言。
```

## 运行全部流程

按顺序自动运行训练 + 评估 + 示例预测：

```bash
python -m src.pipeline --stage all
```

---

## 代码结构说明

```
RumorDetect/
├── data/                              # 数据集
│   ├── train.csv                      # 原始训练集 (2840条)
│   ├── train_cleaned.csv              # 清洗后的训练集 (2791条)
│   ├── val.csv                        # 验证集 (401条)
│   ├── data_conflict_detect.py        # 标签冲突检测脚本
│   └── data_clean.py                  # 数据清洗脚本
│
├── models/                            # 训练产出的模型文件（运行后生成）
│   ├── pretrained/                    # 预训练模型本地路径
│   │   └── bert-large-uncased/        # BERT 预训练模型（自动下载到此目录）
│   ├── bert_rumor_classifier/         # 微调后的BERT模型
│   ├── dense_index.faiss              # FAISS稠密检索索引
│   └── sentence_transformer/          # 嵌入模型缓存
│
├── results/                           # 评估结果（运行后生成）
│   ├── val_predictions.json
│   └── val_predictions.csv
│
├── src/                               # 核心源码
│   ├── __init__.py                    # 模块声明
│   ├── config.py                      # 配置文件（路径、超参数、API配置）
│   ├── data_processor.py              # 数据处理器
│   ├── bert_classifier.py             # BERT分类器模型
│   ├── train.py                       # 训练脚本
│   ├── dense_retriever.py             # 稠密检索器（RAG）
│   ├── llm_explainer.py               # LLM解释生成器
│   ├── inference.py                   # 推理引擎
│   ├── pipeline.py                    # 端到端流水线入口
│   ├── threshold_tune.py              # 阈值调优脚本
│   └── tune.py                        # Optuna超参数搜索脚本
│
├── requirements.txt                   # Python依赖清单
├── .gitignore                         # Git忽略规则
└── README.md                          # 本文件
```

### 核心代码

#### [src/config.py](src/config.py) - 配置文件

- **功能**：集中管理所有路径、超参数、API 配置
- **关键配置项**：BERT 模型名、训练超参数、FAISS 索引路径、LLM API 地址和金钥
- **注意**：API 密钥可通过环境变量 `LLM_API_KEY` 设置，避免硬编码泄露

#### [src/data_processor.py](src/data_processor.py) - 数据处理

- **功能**：
  - 加载 CSV 数据，清洗文本（去除 URL、@用户名，保留#话题标签）
  - 构建 PyTorch DataLoader，支持批处理
  - 自动划分训练/验证集（支持可重复的随机划分）

#### [src/bert_classifier.py](src/bert_classifier.py) - BERT 分类器

- **功能**：
  - `BertRumorClassifier`：BERT → Dropout → Linear 分类头
  - 通过 `output_attentions=True` 获取最后一层注意力权重
  - `get_important_tokens()`：从注意力权重提取模型重点关注的关键词
  - 支持模型保存和加载（BERT 权重 + 分类头分离）
- **注意力提取机制**：作为模型可解释性的一部分，对 [CLS] token 的注意力分数在各注意力头上取平均，排除特殊 token 后取 top-k

#### [src/train.py](src/train.py) - 训练脚本

- **技术**：AdamW 优化器 + 线性学习率预热 + 早停
- **功能**：
  - 完整训练循环，每个 epoch 输出损失和评估指标
  - 使用 Accuracy、F1、Precision、Recall 全面评估
  - 早停机制防止过拟合
  - 训练完成后自动触发 FAISS 索引构建

#### [src/dense_retriever.py](src/dense_retriever.py) - 稠密检索器 (RAG)

- **技术**：sentence-transformers + FAISS (IndexFlatL2)
- **功能**：
  - 使用 `all-MiniLM-L6-v2` 将文本编码为 384 维稠密向量
  - 构建 FAISS L2 距离索引，实现毫秒级相似度检索
  - `retrieve()`：返回 top-k 相似文本及其标签和距离分数
  - `retrieve_formatted()`：将检索结果格式化为 LLM 友好的字符串
  - 支持索引持久化保存和加载

#### [src/llm_explainer.py](src/llm_explainer.py) - LLM 解释生成器

- **技术**：OpenAI 兼容 API 协议 + 请求重试 + 降级方案
- **功能**：
  - 构造系统提示词和用户提示词，引导 LLM 从多角度分析
  - 支持任意兼容 OpenAI 格式的 API（适用于交大 claw 平台）
  - 失败降级策略：API 不可用时自动返回模板化解释
  - 批量处理支持（带进度条）
  - API 请求频率限制：通过类级别的时间戳记录和最小间隔控制（`_min_interval = 6.5s`），确保请求速率 ≤ 10 RPM，避免因频繁调用被 API 网关限流
- **提示词设计**：
  - 系统提示词定义分析维度（语言风格、信息来源、事实核查等）
  - 用户提示词包含推文原文、预测标签、置信度、关键证据、参考案例

#### [src/inference.py](src/inference.py) - 推理引擎

- **技术**：模型加载 + 批量处理 + 结果序列化
- **功能**：
  - `InferenceEngine` 类封装完整推理流程
  - `predict_single()`：单条推文的三阶段推理
  - `predict_batch()`：批量推理（带进度条）
  - `evaluate_on_val()`：完整验证集评估，保存 JSON 和 CSV 结果

#### [src/pipeline.py](src/pipeline.py) - 流水线入口

- **功能**：统一命令行入口，支持 `train` / `eval` / `predict` / `all` / `tune` / `threshold` 六种模式

#### [src/threshold_tune.py](src/threshold_tune.py) - 阈值调优脚本

- **功能**：在验证集上搜索使 F1 最大的谣言概率阈值，解决默认 0.5 阈值不一定是二分类最优的问题
- **技术流程**：
  - 加载已训练的 BERT 分类器，对验证集逐条计算 P(谣言) 概率
  - 在 [0.05, 0.95] 区间内以 0.01 步长遍历，计算每个阈值下的 Accuracy、F1、Precision、Recall
  - 以 F1 为首要指标选出最佳阈值，输出与默认阈值 0.5 的对比
- **输出**：
  - `threshold_tuning.json`：最佳阈值与默认阈值摘要
  - `threshold_curve.csv`：完整阈值-指标曲线
  - `val_probabilities.csv`：每条样本的概率及两种阈值下的预测结果
- **命令行**：`python -m src.threshold_tune --device cuda`

#### [src/tune.py](src/tune.py) - Optuna 超参数搜索脚本

- **技术**：Optuna + TPE 采样器 + Median 剪枝器
- **功能**：
  - 自动搜索 BERT 微调的最佳超参数（batch size、learning rate、weight decay、warmup ratio）
  - 每个 trial 训练少量 epoch，以验证集指定指标（默认 F1）评估参数好坏
  - OOM（显存不足）自动剪枝，避免因 batch size 过大中断搜索
  - 支持断点续调（通过 `--storage sqlite:///results/optuna.db`）
- **搜索空间**：batch size ∈ {16, 32, 64, 128}，lr ∈ [1e-5, 5e-5]，weight_decay ∈ [0, 0.05]，warmup_ratio ∈ [0, 0.2]
- **输出**：
  - `optuna_best_params.json`：最佳参数组合
  - `optuna_trials.csv`：全部 trial 的详细记录
- **命令行**：`python -m src.tune --device cuda --trials 20 --tune-epochs 3`

---