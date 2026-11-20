# Quantum-Inspired Word Embeddings for Text Classification

**Group 14 — Saurabh Tripathi & Vivek Sharma**  
NLP Mini Project (IA1), M.Tech AI&DS, 2026-27
K J Somaiya School of Engineering, Somaiya Vidyavihar University, Mumbai

## Problem statement

Quantum-inspired, complex-valued sentiment classifiers (e.g. ComplexQNN) store two dense embedding
tables (real and imaginary), which take most of the model's memory and grow with the vocabulary.
**Goal:** reduce this embedding storage with a tensor-product (word2ketXS) factorization while
keeping as much classification quality as possible, and measure the trade-off fairly.

## Overview

This project classifies movie-review sentences (SST-2) as **positive** or **negative** with a
quantum-inspired, complex-valued model. Our contribution is a **compressed (factorized) complex
embedding** based on word2ketXS. It stores **117× fewer embedding parameters** than the dense
complex model (15,104 instead of 1,771,520), with about **4.6 points** lower accuracy.

Everything runs on a normal CPU. No GPU or quantum computer is needed.

---

## 1. Setup (one time)

Install Python 3.9 or newer, then in this folder run:

```bash
pip install -r requirements.txt
```

This installs only `numpy`, `autograd` and `matplotlib`.

## 2. Quick demo (no training needed, uses the included checkpoints)

```bash
python src/predict.py "the movie was not good at all"
#  NEGATIVE  (negative  89.3% | positive  10.7%)

python src/predict.py                      # interactive: type sentences one by one
python src/predict.py --model dense "..."  # compare with the existing dense model
python src/app.py                          # small window app: proposed vs existing side by side
python src/explain_compression.py good     # shows step by step how one word's vector is rebuilt
```

## 3. Reproduce the paper's results

| What | Command | Time (laptop CPU) |
|---|---|---|
| One model | `python src/qwe.py --model fact --rank 4 --seed 42` | about 3–5 min |
| All 12 runs + tables + figures | `bash run_all.sh` (Linux/Mac) or `run_all.bat` (Windows) | about 1–2 hours |
| Tables and figures only (from saved results) | `python src/analyze.py` | seconds |

Models: `--model fact` (proposed), `dense` (existing), `real` (real-valued control), `fact_noimag` (ablation).
The code is deterministic: the same seed always gives the same numbers. For example,
`--model fact --rank 4 --seed 13` gives exactly **73.37%** test accuracy.

## 4. Results (test set, mean of seeds 13, 42, 2026)

| Metric | Existing (dense complex) | **Proposed (factorized)** | Real-valued control |
|---|---|---|---|
| Accuracy (%) | 76.99 | 72.43 | 77.76 |
| Precision (%) | 77.06 | 73.14 | 77.81 |
| Recall (%) | 76.99 | 72.45 | 77.76 |
| F1-score (%) | 76.98 | 72.23 | 77.75 |
| Embedding parameters | 1,771,520 | **15,104 (117× fewer)** | 885,760 |
| Total parameters | 1,837,826 | **81,410 (−95.6%)** | 918,914 |
| Peak training memory (MB) | 86.6 | **45.6 (−47.4%)** | 45.4 |
| Training time / epoch (s) | 39.7 | **23.7 (−40.3%)** | 18.9 |

Rank ablation (seed 42): r=2 → 73.20%, r=4 → 71.66%, **r=8 → 75.45%** (still 59× smaller than dense),
dense → 78.25%.

The complete results report (all tables, plots and console output) is `results/Group14_Quantum_Inspired_Word_Embeddings_Results.pdf`.
Full tables are in `results/summary.md` and all figures are in `figures/`.
Accuracy, precision, recall, F1, parameter and memory numbers are identical on every computer.
Training time depends on the computer: the table above shows the paper's run; the re-run included in
`results/` was on a faster machine (23.4 s → 16.0 s per epoch, −31%), so its times are lower but the
proposed model is still faster than the dense one.

## 5. How the compression works (word2ketXS)

1. The 13,840 words are placed in a **118 × 118 grid**, so word *w* gets row `i = w // 118`
   and column `j = w % 118`.
2. Instead of one big 13,840 × 64 table, we store small **row tables and column tables**
   (118 × 8 each), 4 pairs (rank r = 4), separately for the real and imaginary parts.
3. A word's 64 numbers are rebuilt on demand as
   `sum over k of  F1[k][i] ⊗ F2[k][j]` (Kronecker product, 8 × 8 = 64).
4. Storage: dense `2 × 13,840 × 64 = 1,771,520`  vs  factorized `2 × 4 × (118·8 + 118·8) = 15,104`.

The model then reads the sentence with two LSTMs (real and imaginary parts), normalizes the states,
forms a density matrix ρ and scores each class with a measurement-style overlap `|v_c† u_t|²`
(see `forward()` in `src/qwe.py`).

## 6. Folder structure

```
Group14_Quantum_Inspired_Word_Embeddings_Text_Classification/
├── README.md               this file
├── requirements.txt        numpy, autograd, matplotlib
├── run_all.sh / run_all.bat  reproduce every experiment
├── data/SST2/              train.tsv (6,920), dev.tsv (872), test.tsv (1,821)
├── src/
│   ├── qwe.py              model + training (all four models)
│   ├── analyze.py          builds results/summary.md and figures/
│   ├── profile_res.py      memory and CPU measurement
│   ├── sentiment.py        loads a trained model for prediction
│   ├── predict.py          command-line demo
│   ├── app.py              window demo (Tkinter)
│   ├── explain_compression.py  step-by-step compression example
├── checkpoints/            trained models (seed 42) used by the demos
├── configs/                experiment settings (experiments.json)
├── results/                one JSON per run, summary tables, Group14_Quantum_Inspired_Word_Embeddings_Results.pdf, console/ outputs
├── figures/                all result charts (PNG)
└── logs/                   training logs
```

## 7. Data and references

- Dataset: Stanford Sentiment Treebank, binary sentence-level split (SST-2), Socher et al., EMNLP 2013.
  Copy taken from https://github.com/clairett/pytorch-sentiment-classification (`data/SST2`).
- ComplexQNN: Lai, Shi & Chang, *Axioms* 12 (2023) 308.
- Complex-valued LMs: Zhao, Hou & Xu, *Entropy* 24 (2022) 621.
- word2ket: Panahi, Saeedi & Arodz, ICLR 2020.
- Autograd: Maclaurin, Duvenaud & Adams, ICML AutoML Workshop 2015.

## 8. Reproducibility notes

- Random seeds are fixed (`--seed`, default runs use 13, 42 and 2026); runs are fully deterministic on CPU.
- Settings for every experiment are in `configs/experiments.json` and `run_all.sh` / `run_all.bat`.
- Checkpoints in `checkpoints/` are the seed-42 models (the training logs print `models/` because the
  folder was renamed to `checkpoints/` after training; the files are identical).

## 9. Group members

- Saurabh Tripathi
- Vivek Sharma

K J Somaiya School of Engineering (K J Somaiya College of Engineering), Vidyavihar, Mumbai
