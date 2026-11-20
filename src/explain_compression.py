"""Shows, step by step, how the word2ketXS factorization rebuilds one word's 64-number embedding
from two small tables, and counts how much storage it saves.

Usage:  python src/explain_compression.py            (uses word "good")
        python src/explain_compression.py brilliant
"""
import os, sys, math, collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
word = sys.argv[1].lower() if len(sys.argv) > 1 else 'good'

# vocabulary exactly as in training (training sentences only, plus <pad> and <unk>)
cnt = collections.Counter()
for line in open(os.path.join(ROOT, 'data', 'SST2', 'train.tsv'), encoding='utf8'):
    if line.strip():
        cnt.update(line.rsplit('\t', 1)[0].split())
itos = ['<pad>', '<unk>'] + sorted(cnt); stoi = {w: i for i, w in enumerate(itos)}
V, d, r = len(itos), 64, 4
d1 = d2 = 8
v1 = math.ceil(math.sqrt(V)); v2 = math.ceil(V / v1)

w = stoi.get(word, 1)
i, j = w // v2, w % v2
print(f'Vocabulary size V = {V:,} words, embedding size d = {d}')
print(f'Grid: {v1} rows x {v2} columns = {v1*v2:,} slots  (>= {V:,}),   d = {d1} x {d2} = {d}')
print(f'\nWord "{word}" has ID {w}  ->  row i = {w} // {v2} = {i},  column j = {w} % {v2} = {j}')

rng = np.random.RandomState(0)                     # small example tables (random, for illustration)
F1 = rng.normal(0, 0.3, (r, v1, d1)); F2 = rng.normal(0, 0.3, (r, v2, d2))
row = sum(np.kron(F1[k, i], F2[k, j]) for k in range(r))
print(f'\nFor k = 1..{r}: take F1[k][{i}] (8 numbers) and F2[k][{j}] (8 numbers),')
print(f'Kronecker product -> 8 x 8 = 64 numbers, then add the {r} results:')
print('  first 8 numbers of the rebuilt embedding:', np.round(row[:8], 3))
print(f'  length of rebuilt embedding: {row.size}')

dense = 2 * V * d
fact = 2 * r * (v1 * d1 + v2 * d2)
print('\nStorage (real + imaginary channels):')
print(f'  Dense complex      : 2 x {V:,} x {d}            = {dense:>10,} numbers')
print(f'  Factorized (r={r})   : 2 x {r} x ({v1}x{d1} + {v2}x{d2})  = {fact:>10,} numbers')
print(f'  Reduction          : {dense/fact:.0f}x smaller ({100*(1-fact/dense):.2f}% less)')
