# Chinese Word Segmentation and POS Tagging (HMM + CRF)

A Chinese word segmenter built entirely from scratch on a **Hidden Markov Model** — no jieba, no hmmlearn, no sklearn, no PyTorch. The full pipeline (parameter estimation, Viterbi decoding, evaluation) is implemented in ~900 lines of Python with NumPy as the only dependency, and evaluated on the SIGHAN Bakeoff 2005 PKU corpus (People's Daily, 1998).

The repo also contains the supervised counterpart: a **CRF tagger** (`sklearn-crfsuite`) trained for both segmentation and POS tagging, so the two paradigms can be compared on the same task family.

中文版：[README.md](README.md)

## Results

**Part 1 — HMM segmenter (from scratch), SIGHAN 2005 PKU test set:**

| Metric | Score |
|---|---|
| Precision | **80.30%** |
| Recall | **78.89%** |
| F1 | **79.59%** |

For a pure generative HMM with no dictionary features, ~80% F1 on PKU is the expected ceiling — CRF or BiLSTM taggers reach 95%+ precisely by relaxing the two independence assumptions this model is built on (see [Why 80% is the ceiling](#why-80-is-the-ceiling)).

**Part 2 — CRF (sklearn-crfsuite), POS-tagged People's Daily corpus, held-out 20% split:**

| Task | Precision | Recall | F1 / Accuracy |
|---|---|---|---|
| Segmentation (BMES) | 80.26% | 79.74% | **80.00%** |
| POS tagging (character level) | — | — | **87.54%** |

Numbers from the course experiment report. Note the two parts are evaluated on **different corpora** (SIGHAN PKU vs. a locally prepared POS-tagged People's Daily split), so the columns are not directly comparable — they show what each paradigm achieves on its own dataset.

## Why from scratch

Word segmentation is framed as a sequence labelling problem with a 4-tag BMES scheme:

| Tag | Meaning | Example |
|---|---|---|
| B | Begin | 中国 → B, E |
| M | Middle | 共和国 → B, M, E |
| E | End | 中国 → B, E |
| S | Single-character word | 的 → S |

Everything below is implemented by hand in `hmm_segmentation.py`:

- **Parameter estimation** — MLE with Add-1 smoothing for the initial probabilities π, the tag transition matrix A, and the character emission matrix B
- **Viterbi decoding** — dynamic programming over the tag lattice to find the globally most probable tag sequence, then conversion back to word boundaries
- **Evaluation** — precision, recall, F1 and OOV (out-of-vocabulary) rate computed from scratch against the gold standard
- **Model persistence** — trained parameters saved/loaded as JSON (`hmm_model.json`, ~2.8 MB)

## The CRF counterpart

`crf_model.py` trains two `sklearn-crfsuite` models on the same BMES framing:

- **Segmentation CRF** — per-character features: character identity, character type (HAN / NUM / ENG / PUNC), a context window of ±2 characters with their types, plus bigram combination features (`prev_char+char`, `char+next_char`)
- **POS-tagging CRF** — the same feature set, predicting the part-of-speech tag of each character

`prepare_crf_data.py` converts a POS-tagged People's Daily corpus into the CRF training format (`char / BMES / POS / char-type` per line) and does an 80/20 train-test split.

## Quick start

```bash
pip install -r requirements.txt   # numpy, scikit-learn, sklearn-crfsuite
```

**HMM segmenter** (NumPy only):

```python
from hmm_segmentation import ChineseSegmenter

# Train on the PKU corpus (~19,000 sentences)
segmenter = ChineseSegmenter()
segmenter.train("data/pku_training.utf8")
segmenter.save_model("hmm_model.json")

# Segment unseen text
segmenter.load_model("hmm_model.json")
print(" / ".join(segmenter.segment("迈向充满希望的新世纪")))
# 迈向 / 充满 / 希望 / 的 / 新 / 世纪

# Evaluate against the gold standard
segmenter.evaluate("data/pku_test.utf8", "data/pku_test_gold.utf8")
```

Or run the full pipeline (train → segment → evaluate → report):

```bash
python hmm_segmentation.py
```

**CRF tagger:**

```bash
# 1) Put the POS-tagged People's Daily corpus (train_pd.txt, GB18030) next to the script
python prepare_crf_data.py         # generates crf_train.txt / crf_test.txt

# 2) Train both CRF models and print evaluation results
python crf_model.py
```

## Why 80% is the ceiling

A first-order HMM rests on two strong assumptions, and both fail on natural text:

1. **Observation independence** — the current character depends only on the current tag. The model cannot see neighbouring characters, so it loses most contextual cues.
2. **Markov assumption** — the current tag depends only on the previous tag. Long-range dependencies (e.g. "…… 主义" inside a longer named entity) are invisible.

Most importantly, an HMM has no mechanism to represent a **whole word as a unit** — it only ever sees characters through tag emissions. Boundary errors therefore concentrate on OOV words and named entities, which is exactly what the OOV error analysis in this project shows.

The fixes are well known: CRF lets you condition on arbitrary overlapping features; neural taggers learn character embeddings. Those are the natural next steps, and this project exists to make their contribution measurable against a clean baseline.

## Project structure

```
├── hmm_segmentation.py     # HMM: training, Viterbi decoding, evaluation, report
├── crf_model.py            # CRF: segmentation + POS tagging (sklearn-crfsuite)
├── prepare_crf_data.py     # POS-tagged corpus → CRF format, 80/20 split
├── requirements.txt        # numpy, scikit-learn, sklearn-crfsuite
├── data/
│   ├── pku_training.utf8   # PKU training set (~19k sentences, included)
│   ├── pku_test.utf8       # PKU test set (raw text)
│   ├── pku_test_gold.utf8  # PKU test set (gold segmentation)
│   └── icwb2-data/         # full SIGHAN 2005 corpus (~55MB, not in repo, see Data)
├── HMM中文分词实验报告.docx  # course experiment report (Chinese)
└── README.md
```

> `hmm_model.json` (~2.8 MB) is a training artifact — run `python hmm_segmentation.py` to generate it locally.

## Data

[SIGHAN Bakeoff 2005](http://sighan.cs.uchicago.edu/bakeoff2005/) PKU corpus — 1998 People's Daily, ~19,000 training sentences, ~1,900 test sentences. **The three PKU files ship with the repo**, so training and evaluation are reproducible right after cloning.

The full icwb2-data bundle (AS/CityU/MSR/PKU + official scoring scripts, ~55 MB) is **not redistributed** — download it from the SIGHAN page or a public mirror and unpack it into `data/icwb2-data/` for cross-corpus experiments.

The POS-tagged corpus used by `prepare_crf_data.py` (`train_pd.txt`, ~45 MB, GB18030) is **not redistributed** — place it next to the script before running.

The Chinese experiment report is kept as `HMM中文分词实验报告.docx`; the Chinese README is [README.md](README.md).

## License

For academic and learning purposes.
