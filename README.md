# HMM 中文分词与 CRF 词法分析（分词 + 词性标注）

基于**隐马尔可夫模型 (Hidden Markov Model)** 的中文分词系统，全部算法从零手写——不依赖 jieba、hmmlearn、sklearn 等任何 NLP 库，仅用 NumPy。使用 BMES 标注体系，在 SIGHAN 2005 PKU 数据集上训练和评测。

仓库同时包含监督学习对照组：基于 **CRF**（sklearn-crfsuite）的分词与词性标注模型，两种范式可在同一任务上对比。

English version: [README_en.md](README_en.md)

## 实验结果

**第一部分 — HMM 分词（从零实现），SIGHAN 2005 PKU 测试集：**

| 指标 | 分数 |
|---|---|
| 正确率 (Precision) | **80.30%** |
| 召回率 (Recall) | **78.89%** |
| F₁ 值 | **79.59%** |

对于没有任何词典特征的纯生成式 HMM，PKU 上约 80% 的 F1 就是理论上限——CRF 和 BiLSTM 能到 95%+，正是因为它们放宽了本模型赖以成立的两条独立性假设（见 [为什么 80% 是上限](#为什么-80-是上限)）。

**第二部分 — CRF（sklearn-crfsuite），人民日报词性标注语料，80/20 划分的测试集：**

| 任务 | 正确率 | 召回率 | F₁ / 准确率 |
|---|---|---|---|
| 分词 (BMES) | 80.26% | 79.74% | **80.00%** |
| 词性标注（字符级） | — | — | **87.54%** |

数字取自课程实验报告。注意两部分使用**不同语料**（SIGHAN PKU vs. 本地整理的人民日报词性标注语料），列之间不可直接对比，各自反映对应范式在其数据集上的水平。

## 功能特性

**HMM 分词**（`hmm_segmentation.py`，约 900 行，NumPy 之外零依赖）

- 基于词频自动构建词典
- BMES 标注体系（B-词首 / M-词中 / E-词尾 / S-单字词）
- HMM 参数估计：初始概率、转移概率、发射概率，MLE + Add-1 平滑
- Viterbi 算法解码
- 完整的评测指标：正确率、召回率、F₁ 值、OOV（集外词）率
- 模型保存与加载（JSON）

**CRF 对照组**（`crf_model.py`，sklearn-crfsuite）

- 分词 CRF：字符特征（字符本身、字符类型 HAN/NUM/ENG/PUNC、±2 上下文窗口及其类型、bigram 组合特征）
- 词性标注 CRF：同一套特征，预测每个字符的词性标签
- 两个模型训练 + 评测 + 保存（pickle）一条龙

## 项目结构

```
├── hmm_segmentation.py           # HMM 主程序：训练、解码、评测
├── hmm_model.json                # 训练好的 HMM 模型参数（π, A, B）
├── crf_model.py                  # CRF：分词 + 词性标注（sklearn-crfsuite）
├── prepare_crf_data.py           # 词性标注语料 → CRF 格式，80/20 划分
├── requirements.txt              # numpy, scikit-learn, sklearn-crfsuite
├── HMM中文分词实验报告.docx        # 课程实验报告
├── README_en.md                  # English version
├── data/
│   ├── pku_training.utf8         # PKU 训练集
│   ├── pku_test.utf8             # 测试集（原始文本）
│   ├── pku_test_gold.utf8        # 测试集（黄金标准）
│   └── icwb2-data/               # SIGHAN 2005 完整数据集
│       ├── training/             # 各语料库训练数据
│       ├── testing/              # 各语料库测试数据
│       ├── gold/                 # 各语料库标准答案
│       └── scripts/              # 官方评测脚本
└── README.md
```

## 快速开始

### 环境要求

- Python 3.7+

### 安装依赖

```bash
pip install -r requirements.txt   # numpy, scikit-learn, sklearn-crfsuite
```

### HMM 分词

```python
from hmm_segmentation import ChineseSegmenter

# 训练（PKU 语料，约 19,000 句）
segmenter = ChineseSegmenter()
segmenter.train("data/pku_training.utf8")
segmenter.save_model("hmm_model.json")

# 分词
segmenter.load_model("hmm_model.json")
print(" / ".join(segmenter.segment("迈向充满希望的新世纪")))
# 输出: 迈向 / 充满 / 希望 / 的 / 新 / 世纪

# 评测
segmenter.evaluate("data/pku_test.utf8", "data/pku_test_gold.utf8")
```

或一键跑完整流程（训练 → 分词 → 评测 → 报告）：

```bash
python hmm_segmentation.py
```

### CRF 词法分析

```bash
# 1) 把人民日报词性标注语料（train_pd.txt，GB18030 编码）放到脚本同级目录
python prepare_crf_data.py         # 生成 crf_train.txt / crf_test.txt

# 2) 训练两个 CRF 模型并输出评测结果
python crf_model.py
```

## 算法原理

### BMES 标注

| 标签 | 含义 | 示例 |
|------|------|------|
| B | 词首 (Begin) | "中国" → B, E |
| M | 词中 (Middle) | "共和国" → B, M, E |
| E | 词尾 (End) | "中国" → B, E |
| S | 单字词 (Single) | "的" → S |

### HMM 三要素

1. **初始概率 π**：每个标签作为句子开头的概率
2. **转移概率 A**：标签之间的转移概率矩阵
3. **发射概率 B**：标签生成字符的概率

### Viterbi 解码

使用动态规划找到概率最大的标签序列，再根据 BMES 规则还原为分词结果。

### 为什么 80% 是上限

一阶 HMM 建立在两条强假设上，而它们在真实文本上都会失效：

1. **观测独立性**——当前字符只依赖当前标签，模型看不到邻近字符，丢失了大部分上下文线索。
2. **马尔可夫性**——当前标签只依赖前一个标签，长距离依赖（如长命名实体内部的"……主义"）不可见。

最关键的是，HMM 没有任何机制把**整个词作为单元**表示——它只能通过标签发射来"看"字符。因此边界错误集中在 OOV 词和命名实体上，这正是本项目的 OOV 错误分析所显示的。

改进方向是现成的：CRF 可以条件于任意重叠特征，神经序列标注可以学字符嵌入。本项目存在的意义，就是为这些方法的贡献提供一个可度量的干净基线。

## 数据集

- [SIGHAN Bakeoff 2005](http://sighan.cs.uchicago.edu/bakeoff2005/) PKU 语料——1998 年《人民日报》，约 19,000 句训练、约 1,900 句测试。完整 icwb2-data 包（四个语料库 + 官方评测脚本）随仓库附带，方便跨语料实验。
- CRF 部分使用的词性标注语料（`train_pd.txt`，约 45MB，GB18030 编码）**不随仓库分发**——使用前请将其放到脚本同级目录。

## 许可

本项目仅用于学术研究与学习目的。
