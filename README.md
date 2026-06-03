# RumorDetect: 可解释的谣言检测系统

基于 **BERT 微调 + 稠密检索增强 (RAG) + 大语言模型解释生成** 的三阶段复合架构，实现对社交媒体推文的谣言检测与可解释性分析。

## 项目概述

| 组件 | 技术选型 | 功能 |
|------|---------|------|
| **阶段一** | BERT (`bert-base-uncased`) | 推文二分类（谣言/非谣言），提取注意力权重 |
| **阶段二** | sentence-transformers + FAISS | 稠密语义检索，召回训练集中相似案例 |
| **阶段三** | 交大本地 LLM API | 基于预测结果+检索案例，生成自然语言判断依据 |

### 输入/输出

- **输入**：一条英文推文
- **输出1**：二分类标签（0 = 非谣言，1 = 谣言）
- **输出2**：中文自然语言判断依据解释

---

## 环境要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **操作系统** | Windows / Linux / macOS | Linux (Ubuntu 20.04+) |
| **Python** | 3.8+ | 3.10+ |
| **内存** | 8 GB | 16 GB |
| **GPU** | 可选（4GB+ VRAM） | NVIDIA RTX 3060 / 4060 及以上 |
| **CUDA** | 可选 | CUDA 11.7+ |

> **注意**：如果没有 GPU，BERT 训练和推理可以在 CPU 上运行，仅速度较慢（训练约 1 小时，推理每条约 50ms）。

---

## 快速开始

### 1. 克隆或创建项目

```bash
cd RumorDetect
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

如果希望使用 GPU 加速 FAISS，请替换为：

```bash
pip install faiss-gpu
```

### 3. 配置数据

确保 `data/` 目录下包含 `train.csv` 和 `val.csv`，文件格式如下：

| id | text | label | event |
|----|------|-------|-------|
| 536824260615237632 | Swiss museum confirms... | 1 | 0 |

#### 数据清洗

原始训练集存在标签冲突（同一推文被标为不同标签），在训练前通过`data/data_conflict_detect.py`检查。

若发现冲突，人工判断每条冲突的正确标签，通过`data/data_clean.py`生成清洗后的文件`train_cleaned.csv`.

清洗后将 `src/config.py` 中的 `TRAIN_PATH`设置为指向 `train_cleaned.csv`。

### 4. 配置 LLM API

编辑 [src/config.py](src/config.py) 中的 LLM 配置，或者设置环境变量：

```bash
# 方式一：环境变量
export LLM_API_URL="https://models.sjtu.edu.cn/api/v1"
export LLM_API_KEY="your-actual-api-key"
export LLM_MODEL_NAME="qwen3.5-27b"

# Windows PowerShell
$env:LLM_API_URL="https://models.sjtu.edu.cn/api/v1"
$env:LLM_API_KEY="your-actual-api-key"
```

> **提示**：如果未配置 API，系统会使用内置的模板化解释作为降级方案，不影响分类功能。

---

## 训练模型

```bash
# 从项目根目录运行
python -m src.pipeline --stage train
```

训练过程将：

1. **加载数据**：从 `data/train.csv` 读取 2840 条推文
2. **划分数据集**：80% 训练 / 20% 验证
3. **微调 BERT**：使用交叉熵损失，AdamW 优化器，学习率 2e-5
4. **早停机制**：验证集 F1 连续 3 轮不提升即停止
5. **保存模型**：将最佳模型保存至 `models/bert_rumor_classifier/`
6. **构建检索索引**：对全部训练集构建 FAISS 稠密索引，保存至 `models/dense_index.faiss`

### 训练输出

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

## 评估模型

在验证集上运行完整评估（分类 + 解释生成）：

```bash
python -m src.pipeline --stage eval
```

输出结果保存在：

- `results/val_predictions.json`：包含每条推文的完整预测信息
- `results/val_predictions.csv`：易读的表格格式

### 评估输出示例

```
============================================================
RumorDetect - 验证集评估
============================================================
验证集大小: 401 条
加载模型 (设备: cuda)...
加载稠密检索器...
初始化解释生成器...
推理引擎初始化完成！

批量推理: 100%|████████████| 401/401 [02:15<00:00,  2.96it/s]

验证集准确率: 0.8279 (332/401)
结果已保存至: results\val_predictions.json
CSV 结果已保存至: results\val_predictions.csv
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

---

## 运行全部流程

一键运行训练 + 评估 + 示例预测：

```bash
python -m src.pipeline --stage all
```

---

## 代码结构说明

```
RumorDetect/
├── data/                              # 数据集
│   ├── train.csv                      # 原始训练集 (2840条)
│   ├── train_cleaned.csv              # 清洗后的训练集 (2839条)
│   ├── val.csv                        # 验证集 (401条)
│   ├── data_conflict_detect.py        # 标签冲突检测脚本
│   └── data_clean.py                  # 数据清洗脚本
│
├── models/                            # 训练产出的模型文件（运行后生成）
│   ├── pretrained/                    # HuggingFace 预训练模型缓存
│   │   ├── hub/                       #   transformers 模型文件
│   │   └── ...                        #
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
│   └── pipeline.py                    # 端到端流水线入口
│
├── requirements.txt                   # Python依赖清单
├── .gitignore                         # Git忽略规则
└── README.md                          # 本文件
```

### 各模块技术细节

#### [src/config.py](src/config.py) - 配置文件

- **技术**：环境变量 + 硬编码默认值
- **功能**：集中管理所有路径、超参数、API 配置
- **关键配置项**：BERT 模型名、训练超参数、FAISS 索引路径、LLM API 地址和金钥
- **注意**：API 密钥可通过环境变量 `LLM_API_KEY` 设置，避免硬编码泄露

#### [src/data_processor.py](src/data_processor.py) - 数据处理

- **技术**：PyTorch `Dataset` + HuggingFace `Tokenizer`
- **功能**：
  - 加载 CSV 数据，清洗文本（去除 URL、@用户名，保留#话题标签）
  - 构建 PyTorch DataLoader，支持批处理
  - 自动划分训练/验证集（支持可重复的随机划分）

#### [src/bert_classifier.py](src/bert_classifier.py) - BERT 分类器

- **技术**：HuggingFace `AutoModel` + PyTorch `nn.Module`
- **功能**：
  - `BertRumorClassifier`：BERT → Dropout → Linear 分类头
  - 通过 PyTorch forward hook 捕获最后一层注意力权重
  - `get_important_tokens()`：从注意力权重提取模型重点关注的关键词
  - 支持模型保存和加载（BERT 权重 + 分类头分离）
- **注意力提取机制**：对 [CLS] token 的注意力分数在各注意力头上取平均，排除特殊 token 后取 top-k

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

- **功能**：统一命令行入口，支持 `train` / `eval` / `predict` / `all` 四种模式

---

## 架构图

```
输入文本
    │
    ▼
┌────────────────────────────────────┐
│ 阶段一：BERT分类器                    │
│  ├─ bert-base-uncased 编码          │
│  ├─ 线性分类头 → 预测标签(0/1)       │
│  └─ 注意力提取 → 关键证据词          │
└───────────┬────────────────────────┘
            │ 预测标签 + 置信度 + 关键证据
            ▼
┌────────────────────────────────────┐
│ 阶段二：稠密检索 (RAG)               │
│  ├─ sentence-transformers 编码      │
│  ├─ FAISS 索引检索 top-k 相似案例    │
│  └─ 格式化检索结果                   │
└───────────┬────────────────────────┘
            │ 相似案例文本 + 真实标签
            ▼
┌────────────────────────────────────┐
│ 阶段三：LLM 解释生成                  │
│  ├─ 构造 System + User Prompt      │
│  ├─ 调用交大本地 API                │
│  └─ 输出中文判断依据                 │
└───────────┬────────────────────────┘
            │
            ▼
    ┌───────────────┐
    │ 输出:           │
    │ 标签 + 解释文本  │
    └───────────────┘
```

---

## API 配置指南

### 环境变量方式（推荐）

```bash
# Linux / macOS
export LLM_API_URL="https://models.sjtu.edu.cn/api/v1"
export LLM_API_KEY="your-key-here"
export LLM_MODEL_NAME="qwen3.5-27b"

# Windows
set LLM_API_URL=https://models.sjtu.edu.cn/api/v1
set LLM_API_KEY=your-key-here
set LLM_MODEL_NAME=qwen3.5-27b
```

### 配置文件方式

直接编辑 [src/config.py](src/config.py) 中的以下变量：

```python
LLM_API_URL = "https://claw.sjtu.edu.cn/api/v1/chat/completions"
LLM_API_KEY = "your-actual-api-key"
LLM_MODEL_NAME = "qwen3.5-27b"
```

---

## 项目集成

本系统的模块化设计使得各组件可以独立使用：

```python
# 仅使用 BERT 分类
from src.bert_classifier import BertRumorClassifier, load_model
model, tokenizer = load_model("models/bert_rumor_classifier")

# 仅使用 RAG 检索
from src.dense_retriever import DenseRetriever
retriever = DenseRetriever()
results = retriever.retrieve("your query text")

# 仅使用 LLM 解释生成
from src.llm_explainer import LLMExplainer
explainer = LLMExplainer()
explanation = explainer.generate(text, prediction=1, confidence=0.85, ...)
```