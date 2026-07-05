# HMM 中文分词

基于**隐马尔可夫模型 (Hidden Markov Model)** 的中文分词系统，使用 BMES 标注体系，在 SIGHAN 2005 PKU 数据集上训练和评测。

## 功能特性

- 基于词频自动构建词典
- BMES 标注体系（B-词首 / M-词中 / E-词尾 / S-单字词）
- HMM 参数估计（初始概率、转移概率、发射概率）
- Viterbi 算法解码
- 完整的评测指标：正确率 (Precision)、召回率 (Recall)、F₁ 值 (F-Measure)
- 支持 OOV（集外词）检测
- 模型保存与加载

## 项目结构

```
├── hmm_segmentation.py           # 主程序：训练、分词、评测
├── hmm_model.json                # 训练好的 HMM 模型参数
├── HMM中文分词实验报告.docx        # 实验报告
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
- NumPy

### 安装依赖

```bash
pip install numpy
```

### 训练模型

```python
from hmm_segmentation import ChineseSegmenter

segmenter = ChineseSegmenter()
segmenter.train("data/pku_training.utf8")
segmenter.save_model("hmm_model.json")
```

### 分词示例

```python
from hmm_segmentation import ChineseSegmenter

segmenter = ChineseSegmenter()
segmenter.load_model("hmm_model.json")

text = "迈向充满希望的新世纪"
result = segmenter.segment(text)
print(" / ".join(result))
# 输出: 迈向 / 充满 / 希望 / 的 / 新 / 世纪
```

### 运行评测

```python
from hmm_segmentation import ChineseSegmenter

segmenter = ChineseSegmenter()
segmenter.load_model("hmm_model.json")

precision, recall, f1, oov_rate = segmenter.evaluate(
    "data/pku_test.utf8",
    "data/pku_test_gold.utf8"
)
print(f"正确率: {precision:.2%}")
print(f"召回率: {recall:.2%}")
print(f"F₁值:   {f1:.2%}")
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

## 数据集

使用 [SIGHAN 2005 中文分词评测](http://sighan.cs.uchicago.edu/bakeoff2005/) 的 PKU 数据集：

- 训练集：约 19,000 句
- 测试集：约 1,900 句
- 语料来源：1998 年《人民日报》

## 许可

本项目仅用于学术研究与学习目的。
