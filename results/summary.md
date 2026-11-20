# Results (test set, mean ± std over seeds 13, 42, 2026)

| Metric | Existing (dense complex) | Proposed (factorized) | Improvement | Real-valued control |
|---|---|---|---|---|
| Accuracy (%) | 76.99 ± 1.35 | 72.43 ± 0.86 | -4.56 pp | 77.76 ± 2.28 |
| Precision (%) | 77.06 ± 1.38 | 73.14 ± 0.49 | -3.92 pp | 77.81 ± 2.22 |
| Recall (%) | 76.99 ± 1.35 | 72.45 ± 0.86 | -4.54 pp | 77.76 ± 2.28 |
| F1-score (%) | 76.98 ± 1.35 | 72.23 ± 0.98 | -4.75 pp | 77.75 ± 2.30 |
| Specificity (%) | 77.49 ± 4.28 | 64.07 ± 3.71 | -13.41 pp | 77.19 ± 4.86 |
| Error rate (%) | 23.01 ± 1.35 | 27.57 ± 0.86 | +4.56 pp | 22.24 ± 2.28 |
| False positive rate (%) | 22.51 ± 4.28 | 35.93 ± 3.71 | +13.41 pp | 22.81 ± 4.86 |
| Embedding parameters | 1,771,520 | 15,104 | 117x fewer (-99.15%) | 885,760 |
| Total parameters | 1,837,826 | 81,410 | -95.6% | 918,914 |
| Embedding storage (KB, float32) | 6920.0 | 59.0 | -99.15% | 3460.0 |
| Training time / epoch (s) | 23.4 | 16.0 | -31.4% | 11.8 |
| Inference latency (ms/sentence) | 1.54 | 1.60 | +3.8% | 0.73 |
| Peak training memory (MB) | 86.6 | 45.6 | -47.4% | 45.4 |
| CPU time / epoch, profiled (s) | 70.8 | 68.7 | -2.9% | 34.1 |

## Accuracy per seed (%)

| Model | 13 | 42 | 2026 |
|---|---|---|---|
| Dense complex (existing) | 77.16 | 78.25 | 75.56 |
| Factorized complex (proposed) | 73.37 | 71.66 | 72.27 |
| Real-valued control | 79.02 | 79.13 | 75.12 |

## Ablation (seed 42)

| Configuration | Embedding params | Val macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---|---|---|
| Factorized r=2 | 7,552 | 73.99 | 73.20 | 73.18 |
| Factorized r=4 (proposed) | 15,104 | 74.01 | 71.66 | 71.37 |
| Factorized r=8 | 30,208 | 77.36 | 75.45 | 75.35 |
| Factorized r=4, no imaginary | 7,552 | 74.56 | 70.68 | 70.56 |
| Dense complex (existing) | 1,771,520 | 79.35 | 78.25 | 78.23 |
