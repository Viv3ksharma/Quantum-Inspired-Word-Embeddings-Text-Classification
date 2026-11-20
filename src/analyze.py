"""Builds the result tables and figures of the paper from results/*.json.

Usage:  python src/analyze.py
Outputs: results/summary.md, results/summary.csv, figures/*.png
"""
import os, json, glob, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results'); FIG = os.path.join(ROOT, 'figures'); os.makedirs(FIG, exist_ok=True)
runs = [json.load(open(f)) for f in glob.glob(os.path.join(R, '*_s*.json'))]

def get(model, rank=None, seed=None):
    return [r for r in runs if r['model'] == model and (rank is None or r['rank'] == rank) and (seed is None or r['seed'] == seed)]

ORDER = ['dense', 'fact', 'real']
MAIN = {'dense': get('dense'), 'fact': get('fact', 4), 'real': get('real')}
for k, v in MAIN.items():
    if len(v) != 3:
        raise SystemExit(f'Expected 3 seeds for {k}, found {len(v)}. Run run_all first.')
NAME = {'dense': 'Dense complex (existing)', 'fact': 'Factorized complex (proposed)', 'real': 'Real-valued control'}
TICK = {'dense': 'Existing:\nDense complex', 'fact': 'Proposed:\nFactorized', 'real': 'Control:\nReal-valued'}
COL = {'dense': '#4A78A8', 'fact': '#17375E', 'real': '#9DB9DB'}

def ms(v): return float(np.mean(v)), float(np.std(v, ddof=1))
S = {}
for m, rs in MAIN.items():
    d = {k: ms([100 * r['test'][k] for r in rs]) for k in ['acc', 'prec', 'rec', 'f1', 'spec', 'err', 'fpr', 'f1_neg', 'f1_pos']}
    d['tpe'] = ms([r['time_per_epoch'] for r in rs]); d['lat'] = ms([r['latency_ms'] for r in rs])
    d['ep'] = ms([r['epochs_run'] for r in rs])
    d['params'] = rs[0]['n_params']; d['emb'] = rs[0]['emb_params']; d['embkb'] = rs[0]['emb_kb']
    d['cm'] = np.sum([np.array(r['test']['cm']) for r in rs], 0)
    d['seeds'] = {r['seed']: 100 * r['test']['acc'] for r in rs}
    S[m] = d
PR = {}
for m in ORDER:
    f = os.path.join(R, f'profile_{m}.json')
    if os.path.exists(f): PR[m] = json.load(open(f))

# ---------------- tables ----------------
D, F, Re = S['dense'], S['fact'], S['real']
rows = [('Accuracy (%)', 'acc'), ('Precision (%)', 'prec'), ('Recall (%)', 'rec'), ('F1-score (%)', 'f1'),
        ('Specificity (%)', 'spec'), ('Error rate (%)', 'err'), ('False positive rate (%)', 'fpr')]
md = ['# Results (test set, mean ± std over seeds 13, 42, 2026)\n',
      '| Metric | Existing (dense complex) | Proposed (factorized) | Improvement | Real-valued control |',
      '|---|---|---|---|---|']
csvrows = [['metric', 'dense', 'fact', 'real', 'improvement_fact_minus_dense']]
for lab, k in rows:
    md.append(f'| {lab} | {D[k][0]:.2f} ± {D[k][1]:.2f} | {F[k][0]:.2f} ± {F[k][1]:.2f} | {F[k][0]-D[k][0]:+.2f} pp | {Re[k][0]:.2f} ± {Re[k][1]:.2f} |')
    csvrows.append([lab, f'{D[k][0]:.2f}', f'{F[k][0]:.2f}', f'{Re[k][0]:.2f}', f'{F[k][0]-D[k][0]:+.2f}'])
md.append(f"| Embedding parameters | {D['emb']:,} | {F['emb']:,} | {D['emb']/F['emb']:.0f}x fewer (-{100*(1-F['emb']/D['emb']):.2f}%) | {Re['emb']:,} |")
md.append(f"| Total parameters | {D['params']:,} | {F['params']:,} | -{100*(1-F['params']/D['params']):.1f}% | {Re['params']:,} |")
md.append(f"| Embedding storage (KB, float32) | {D['embkb']:.1f} | {F['embkb']:.1f} | -{100*(1-F['embkb']/D['embkb']):.2f}% | {Re['embkb']:.1f} |")
md.append(f"| Training time / epoch (s) | {D['tpe'][0]:.1f} | {F['tpe'][0]:.1f} | -{100*(1-F['tpe'][0]/D['tpe'][0]):.1f}% | {Re['tpe'][0]:.1f} |")
md.append(f"| Inference latency (ms/sentence) | {D['lat'][0]:.2f} | {F['lat'][0]:.2f} | {100*(F['lat'][0]/D['lat'][0]-1):+.1f}% | {Re['lat'][0]:.2f} |")
if len(PR) == 3:
    md.append(f"| Peak training memory (MB) | {PR['dense']['peak_train_mb']:.1f} | {PR['fact']['peak_train_mb']:.1f} | -{100*(1-PR['fact']['peak_train_mb']/PR['dense']['peak_train_mb']):.1f}% | {PR['real']['peak_train_mb']:.1f} |")
    md.append(f"| CPU time / epoch, profiled (s) | {PR['dense']['cpu_train_s']:.1f} | {PR['fact']['cpu_train_s']:.1f} | -{100*(1-PR['fact']['cpu_train_s']/PR['dense']['cpu_train_s']):.1f}% | {PR['real']['cpu_train_s']:.1f} |")
md += ['', '## Accuracy per seed (%)', '', '| Model | 13 | 42 | 2026 |', '|---|---|---|---|']
for m in ORDER:
    md.append(f"| {NAME[m]} | {S[m]['seeds'][13]:.2f} | {S[m]['seeds'][42]:.2f} | {S[m]['seeds'][2026]:.2f} |")
md += ['', '## Ablation (seed 42)', '', '| Configuration | Embedding params | Val macro-F1 | Test accuracy | Test macro-F1 |', '|---|---|---|---|---|']
abl = [('Factorized r=2', get('fact', 2, 42)), ('Factorized r=4 (proposed)', get('fact', 4, 42)), ('Factorized r=8', get('fact', 8, 42)),
       ('Factorized r=4, no imaginary', get('fact_noimag', 4, 42)), ('Dense complex (existing)', get('dense', None, 42))]
abl = [(n, x[0]) for n, x in abl if x]
for n, r in abl:
    md.append(f"| {n} | {r['emb_params']:,} | {100*r['best_val_f1']:.2f} | {100*r['test']['acc']:.2f} | {100*r['test']['f1']:.2f} |")
open(os.path.join(R, 'summary.md'), 'w').write('\n'.join(md) + '\n')
with open(os.path.join(R, 'summary.csv'), 'w', newline='') as fh: csv.writer(fh).writerows(csvrows)

# ---------------- figures ----------------
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
def save(fig, name): fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=200, bbox_inches='tight'); plt.close(fig)

def bar(k, title, fname, lo, hi, ylabel='Performance (%)'):
    fig, ax = plt.subplots(figsize=(6.4, 3.6)); xs = np.arange(3)
    v = [S[m][k][0] for m in ORDER]; s = [S[m][k][1] for m in ORDER]
    ax.bar(xs, v, 0.55, color=[COL[m] for m in ORDER], yerr=s, capsize=4)
    for x, a, b in zip(xs, v, s): ax.text(x, a + b + (hi - lo) * 0.02, f'{a:.2f}%\n±{b:.2f}', ha='center', fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels([TICK[m] for m in ORDER]); ax.set_ylim(lo, hi); ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', color='#17375E'); ax.yaxis.grid(True, color='#e6e6e6'); ax.set_axisbelow(True)
    save(fig, fname)
bar('acc', 'Accuracy Comparison', 'accuracy.png', 60, 85)
bar('prec', 'Precision Comparison (macro)', 'precision.png', 60, 85)
bar('rec', 'Recall Comparison (macro)', 'recall.png', 60, 85)
bar('f1', 'F1-Score Comparison (macro)', 'f1.png', 60, 85)
bar('err', 'Error Rate Comparison', 'error_rate.png', 0, 40, 'Error rate (%)')

fig, ax = plt.subplots(figsize=(7.2, 3.8)); mets = [('acc', 'Accuracy'), ('prec', 'Precision'), ('rec', 'Recall'), ('f1', 'F1-score'), ('spec', 'Specificity')]
xs = np.arange(len(mets)); w = 0.26
for i, m in enumerate(ORDER):
    ax.bar(xs + (i - 1) * w, [S[m][k][0] for k, _ in mets], w * 0.92, yerr=[S[m][k][1] for k, _ in mets], capsize=2.5, color=COL[m], label=NAME[m])
ax.set_xticks(xs); ax.set_xticklabels([n for _, n in mets]); ax.set_ylim(55, 85); ax.set_ylabel('Performance (%)')
ax.legend(frameon=False, ncol=3, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.13)); ax.yaxis.grid(True, color='#e6e6e6'); ax.set_axisbelow(True)
save(fig, 'grouped_metrics.png')

fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2))
for ax, (k, t, yl, log) in zip(axs, [('emb', 'Embedding parameters', 'Parameters (log)', True), ('tpe', 'Training time / epoch', 'Seconds', False), ('lat', 'Inference latency', 'ms / sentence', False)]):
    v = [S[m][k] if not isinstance(S[m][k], tuple) else S[m][k][0] for m in ORDER]
    ax.bar(range(3), v, 0.55, color=[COL[m] for m in ORDER])
    if log: ax.set_yscale('log'); ax.set_ylim(5e3, 5e6)
    for x, a in enumerate(v): ax.text(x, a * (1.15 if log else 1.03), f'{a:,.0f}' if k == 'emb' else f'{a:.2f}', ha='center', fontsize=8.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(['Dense\ncomplex', 'Factorized\ncomplex', 'Real-\nvalued'], fontsize=8.5)
    ax.set_title(t, fontweight='bold', color='#17375E'); ax.set_ylabel(yl)
    if not log: ax.set_ylim(0, max(v) * 1.25)
save(fig, 'cost.png')

if len(PR) == 3:
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for ax, (k, t, yl) in zip(axs, [('peak_train_mb', 'Peak memory: training', 'MB'), ('rss_mb', 'Peak process memory', 'MB'), ('cpu_train_s', 'CPU time per epoch', 'CPU seconds')]):
        v = [PR[m][k] for m in ORDER]
        ax.bar(range(3), v, 0.55, color=[COL[m] for m in ORDER])
        for x, a in enumerate(v): ax.text(x, a * 1.02, f'{a:.1f}', ha='center', fontsize=8.5)
        ax.set_xticks(range(3)); ax.set_xticklabels(['Dense\ncomplex', 'Factorized\ncomplex', 'Real-\nvalued'], fontsize=8.5)
        ax.set_title(t, fontweight='bold', color='#17375E'); ax.set_ylabel(yl)
    save(fig, 'resource_usage.png')

fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.1))
for ax, m in zip(axs, ORDER):
    cm = S[m]['cm']; ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max() * 1.15)
    for i in range(2):
        for j in range(2): ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', fontsize=11, color='white' if cm[i, j] > cm.max() * 0.6 else '#17375E')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(['Neg', 'Pos']); ax.set_yticklabels(['Neg', 'Pos'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(NAME[m], fontsize=9.5, fontweight='bold', color='#17375E')
save(fig, 'confusion_matrices.png')

rk = [(r['emb_params'], 100 * r['test']['acc'], n) for n, r in abl if n.startswith('Factorized r=') and 'no imag' not in n]
if rk:
    fig, ax = plt.subplots(figsize=(6.2, 3.4)); rk.sort()
    ax.plot([a for a, _, _ in rk], [b for _, b, _ in rk], '-o', color='#17375E', lw=2, ms=8)
    for a, b, n in rk:
        off = {'Factorized r=2': (-6, 12), 'Factorized r=4': (30, -28), 'Factorized r=8': (0, 12)}.get(n.split(' (')[0], (0, 10))
        ax.annotate(f'{n.split(" (")[0]}\n{b:.2f}%', (a, b), textcoords='offset points', xytext=off, ha='center', fontsize=8.5)
    dn = [r for n, r in abl if n.startswith('Dense')]
    if dn:
        ax.plot(dn[0]['emb_params'], 100 * dn[0]['test']['acc'], 's', color='#4A78A8', ms=9)
        ax.annotate(f"Dense\n{100*dn[0]['test']['acc']:.2f}%", (dn[0]['emb_params'], 100 * dn[0]['test']['acc']), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=8.5)
    ax.set_xscale('log'); ax.set_xlim(3500, 4e6); ax.set_xlabel('Embedding parameters (log)'); ax.set_ylabel('Test accuracy (%)'); ax.set_ylim(68, 82)
    ax.set_title('Rank ablation: storage vs accuracy (seed 42)', fontweight='bold', color='#17375E')
    save(fig, 'rank_ablation.png')

fig, ax = plt.subplots(figsize=(6.2, 3.3))
for m in ORDER:
    r = [x for x in MAIN[m] if x['seed'] == 42][0]
    ax.plot([e['epoch'] for e in r['hist']], [100 * e['val_f1'] for e in r['hist']], '-o', ms=4, lw=2, color=COL[m], label=NAME[m])
ax.set_xlabel('Epoch'); ax.set_ylabel('Validation macro-F1 (%)'); ax.legend(frameon=False, fontsize=8.5)
ax.set_title('Validation F1 per epoch (seed 42)', fontweight='bold', color='#17375E')
save(fig, 'training_curves.png')

print(open(os.path.join(R, 'summary.md')).read())
print('Figures written to', FIG)
