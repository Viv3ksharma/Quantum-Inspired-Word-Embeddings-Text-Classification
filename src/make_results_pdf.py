"""Builds results/Group14_Quantum_Inspired_Word_Embeddings_Results.pdf (all tables, plots and real values) from results/*.json and figures/.
Needs XeLaTeX.  Usage:  python src/make_results_pdf.py
"""
import os, json, glob, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results'); FIG = os.path.join(ROOT, 'figures')
runs = [json.load(open(f)) for f in glob.glob(os.path.join(R, '*_s*.json'))]
def get(m, rank=None, seed=None):
    return [r for r in runs if r['model'] == m and (rank is None or r['rank'] == rank) and (seed is None or r['seed'] == seed)]
ORDER = ['dense', 'fact', 'real']
MAIN = {'dense': get('dense'), 'fact': get('fact', 4), 'real': get('real')}
NAME = {'dense': 'Existing: dense complex', 'fact': 'Proposed: factorized complex', 'real': 'Control: real-valued'}
PR = {m: json.load(open(os.path.join(R, f'profile_{m}.json'))) for m in ORDER}

def ms(v): return float(np.mean(v)), float(np.std(v, ddof=1))
S = {}
for m, rs in MAIN.items():
    S[m] = {k: ms([100 * r['test'][k] for r in rs]) for k in ['acc', 'prec', 'rec', 'f1', 'spec', 'err', 'fpr', 'f1_neg', 'f1_pos']}
    S[m].update(tpe=ms([r['time_per_epoch'] for r in rs]), lat=ms([r['latency_ms'] for r in rs]), ep=ms([r['epochs_run'] for r in rs]),
                val=ms([100 * r['best_val_f1'] for r in rs]), params=rs[0]['n_params'], emb=rs[0]['emb_params'], embkb=rs[0]['emb_kb'],
                cm=np.sum([np.array(r['test']['cm']) for r in rs], 0), seed={r['seed']: r for r in rs})
D, F, Rl = S['dense'], S['fact'], S['real']
cfg = MAIN['fact'][0]

def pm(x): return f'{x[0]:.2f} $\\pm$ {x[1]:.2f}'
def sg(x): return f'{x:+.2f}'.replace('-', '$-$')
T = []
T.append(r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.1cm]{geometry}
\usepackage{fontspec}\setmainfont{TeX Gyre Termes}\setsansfont{TeX Gyre Heros}
\usepackage{booktabs,graphicx,float,xcolor,array,caption,fancyhdr,hyperref,longtable}\usepackage{fvextra}\fvset{breaklines=true,fontsize=\footnotesize,frame=single,rulecolor=\color{gray}}
\definecolor{maroon}{RGB}{140,22,22}
\captionsetup{font=small,labelfont=bf}
\pagestyle{fancy}\fancyhf{}\lhead{\small Quantum-Inspired Word Embeddings --- Results}\rhead{\small Group 14}\cfoot{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\newcommand{\sect}[1]{\section*{\color{maroon}#1}}
\begin{document}
\begin{center}
{\LARGE\bfseries\color{maroon} Results Report}\\[4pt]
{\Large Quantum-Inspired Word Embeddings for Text Classification}\\[8pt]
{\large Saurabh Tripathi \quad Vivek Sharma \quad (Group 14)}\\[2pt]
K J Somaiya School of Engineering, Somaiya Vidyavihar University, Mumbai
\end{center}
\vspace{4pt}\hrule\vspace{8pt}
""")
T.append(r"""\sect{1. Experimental setup}
\noindent\begin{tabular}{@{}>{\bfseries}p{0.17\textwidth}p{0.78\textwidth}@{}}
\toprule
Dataset & SST-2 (binary, sentence level): 6,920 train / 872 validation / 1,821 test \\""" + f"""
Vocabulary & {cfg['V']:,} tokens (training text only); max length {cfg['MAXL']} tokens; {100*cfg['oov_test']:.2f}\\% unknown test tokens \\\\
Models & Existing: dense complex embedding; Proposed: factorized complex (word2ketXS, $r=4$); Control: real-valued \\\\
Shared settings & $d=h=64$, batch 32, Adam ($\\eta=10^{{-3}}$), $\\tau=0.1$, max 20 epochs, early stopping (patience 4) \\\\
Factorization & grid ${cfg['v1']}\\times{cfg['v2']}$, $d={cfg['d1']}\\times{cfg['d2']}$, rank $r=4$, separately for real and imaginary parts \\\\
Seeds & 13, 42, 2026 (values below are mean $\\pm$ standard deviation over the three seeds) \\\\
Hardware & CPU only, single thread, NumPy + autograd \\\\
\\bottomrule
\\end{{tabular}}
""")
T.append(r"""
\sect{2. Main results (test set, 1,821 sentences)}
\begin{table}[H]\centering\caption{Classification metrics in \% (mean $\pm$ s.d. over 3 seeds). Improvement = proposed $-$ existing (percentage points).}
\begin{tabular}{@{}lcccc@{}}\toprule
Metric & Existing (dense) & \textbf{Proposed (factorized)} & Improvement & Control (real) \\\midrule
""")
for lab, k in [('Accuracy', 'acc'), ('Precision (macro)', 'prec'), ('Recall (macro)', 'rec'), ('F1-score (macro)', 'f1'),
               ('Specificity', 'spec'), ('Error rate', 'err'), ('False positive rate', 'fpr'), ('F1 negative class', 'f1_neg'), ('F1 positive class', 'f1_pos')]:
    T.append(f"{lab} & {pm(D[k])} & {pm(F[k])} & {sg(F[k][0]-D[k][0])} & {pm(Rl[k])} \\\\\n")
T.append(r"""\bottomrule\end{tabular}\end{table}
\begin{table}[H]\centering\caption{Test accuracy (\%) per seed.}
\begin{tabular}{@{}lcccc@{}}\toprule Model & Seed 13 & Seed 42 & Seed 2026 & Mean \\\midrule
""")
for m in ORDER:
    a = [100 * S[m]['seed'][s]['test']['acc'] for s in (13, 42, 2026)]
    T.append(f"{NAME[m]} & {a[0]:.2f} & {a[1]:.2f} & {a[2]:.2f} & {np.mean(a):.2f} \\\\\n")
T.append(r"""\bottomrule\end{tabular}\end{table}
\begin{table}[H]\centering\caption{Confusion matrices on the test set, summed over the three seeds (rows = actual, columns = predicted).}
\begin{tabular}{@{}lcccc@{}}\toprule Model & Neg$\to$Neg (TN) & Neg$\to$Pos (FP) & Pos$\to$Neg (FN) & Pos$\to$Pos (TP) \\\midrule
""")
for m in ORDER:
    c = S[m]['cm']; T.append(f"{NAME[m]} & {c[0,0]:,} & {c[0,1]:,} & {c[1,0]:,} & {c[1,1]:,} \\\\\n")
T.append(r"\bottomrule\end{tabular}\end{table}" + "\n")

T.append(r"""\sect{3. Model size, time and resource usage}
\begin{table}[H]\centering\caption{Model size and computation cost (mean over 3 seeds).}
\begin{tabular}{@{}lccc@{}}\toprule Quantity & Existing (dense) & \textbf{Proposed} & Control (real) \\\midrule
""")
T.append(f"Embedding parameters & {D['emb']:,} & \\textbf{{{F['emb']:,}}} & {Rl['emb']:,} \\\\\n")
T.append(f"Total parameters & {D['params']:,} & \\textbf{{{F['params']:,}}} & {Rl['params']:,} \\\\\n")
T.append(f"Embedding storage (KB, float32) & {D['embkb']:.1f} & \\textbf{{{F['embkb']:.1f}}} & {Rl['embkb']:.1f} \\\\\n")
T.append(f"Training time / epoch (s) & {pm(D['tpe'])} & {pm(F['tpe'])} & {pm(Rl['tpe'])} \\\\\n")
T.append(f"Epochs until early stop & {D['ep'][0]:.1f} & {F['ep'][0]:.1f} & {Rl['ep'][0]:.1f} \\\\\n")
T.append(f"Inference latency (ms / sentence) & {pm(D['lat'])} & {pm(F['lat'])} & {pm(Rl['lat'])} \\\\\n")
T.append(f"Peak training memory (MB) & {PR['dense']['peak_train_mb']:.1f} & \\textbf{{{PR['fact']['peak_train_mb']:.1f}}} & {PR['real']['peak_train_mb']:.1f} \\\\\n")
T.append(f"Peak inference memory (MB) & {PR['dense']['peak_inf_mb']:.1f} & {PR['fact']['peak_inf_mb']:.1f} & {PR['real']['peak_inf_mb']:.1f} \\\\\n")
T.append(f"Peak process memory, RSS (MB) & {PR['dense']['rss_mb']:.1f} & {PR['fact']['rss_mb']:.1f} & {PR['real']['rss_mb']:.1f} \\\\\n")
T.append(f"CPU time / profiled epoch (s) & {PR['dense']['cpu_train_s']:.1f} & {PR['fact']['cpu_train_s']:.1f} & {PR['real']['cpu_train_s']:.1f} \\\\\n")
T.append(f"CPU utilization (\\%) & {PR['dense']['cpu_util_train']:.1f} & {PR['fact']['cpu_util_train']:.1f} & {PR['real']['cpu_util_train']:.1f} \\\\\n")
T.append(r"\bottomrule\end{tabular}\end{table}" + "\n")
T.append(f"""\\noindent Storage of the proposed embedding: dense $2\\times{cfg['V']:,}\\times64={D['emb']:,}$ values versus factorized
$2\\times4\\times({cfg['v1']}\\cdot8+{cfg['v2']}\\cdot8)={F['emb']:,}$ values, i.e.\\ \\textbf{{{D['emb']/F['emb']:.0f}$\\times$ fewer}}
($-${100*(1-F['emb']/D['emb']):.2f}\\%). Total parameters fall by {100*(1-F['params']/D['params']):.1f}\\%, peak training memory by
{100*(1-PR['fact']['peak_train_mb']/PR['dense']['peak_train_mb']):.1f}\\% and training time per epoch by {100*(1-F['tpe'][0]/D['tpe'][0]):.1f}\\%.
Timing values depend on the machine; accuracy values are exactly reproducible with the same seed.
""")

T.append(r"""\sect{4. Ablation study (seed 42)}
\begin{table}[H]\centering\caption{Effect of the factorization rank and of the imaginary channel.}
\begin{tabular}{@{}lrccc@{}}\toprule Configuration & Embedding params & Val.\ macro-F1 & Test accuracy & Test macro-F1 \\\midrule
""")
for n, sel in [('Factorized, $r=2$', get('fact', 2, 42)), ('Factorized, $r=4$ (proposed)', get('fact', 4, 42)), ('Factorized, $r=8$', get('fact', 8, 42)),
               ('Factorized, $r=4$, no imaginary channel', get('fact_noimag', 4, 42)), ('Dense complex (existing)', get('dense', None, 42)), ('Real-valued control', get('real', None, 42))]:
    r = sel[0]; T.append(f"{n} & {r['emb_params']:,} & {100*r['best_val_f1']:.2f} & {100*r['test']['acc']:.2f} & {100*r['test']['f1']:.2f} \\\\\n")
T.append(r"\bottomrule\end{tabular}\end{table}" + "\n")

T.append(r"""\sect{5. All individual runs (raw values)}
\begin{longtable}{@{}lrrrrrrr@{}}
\caption{Every training run: test metrics in \%, epochs run and training time.}\\\toprule
Run & Acc. & Prec. & Rec. & F1 & Spec. & Epochs & Time (s) \\\midrule\endhead
""")
for r in sorted(runs, key=lambda r: (r['model'], r['rank'] or 0, r['seed'])):
    tag = r['model'] + (f" r={r['rank']}" if r['rank'] else '') + f" seed {r['seed']}"
    t = r['test']
    T.append(f"{tag.replace('_', chr(92)+'_')} & {100*t['acc']:.2f} & {100*t['prec']:.2f} & {100*t['rec']:.2f} & {100*t['f1']:.2f} & {100*t['spec']:.2f} & {r['epochs_run']} & {r['train_time_s']:.0f} \\\\\n")
T.append(r"\bottomrule\end{longtable}" + "\n")
figs = [('accuracy.png', 'Accuracy comparison (error bars: s.d. over seeds).', 0.68),
        ('precision.png', 'Macro precision comparison.', 0.68), ('recall.png', 'Macro recall comparison.', 0.68),
        ('f1.png', 'Macro F1-score comparison.', 0.68), ('grouped_metrics.png', 'All classification metrics side by side.', 0.85),
        ('error_rate.png', 'Error rate comparison.', 0.68), ('confusion_matrices.png', 'Confusion matrices (summed over three seeds).', 0.98),
        ('cost.png', 'Embedding parameters (log scale), training time per epoch and inference latency.', 0.98),
        ('resource_usage.png', 'Peak training memory, peak process memory and CPU time per epoch.', 0.98),
        ('rank_ablation.png', 'Rank ablation: storage versus accuracy (seed 42).', 0.75),
        ('training_curves.png', 'Validation macro-F1 per training epoch (seed 42).', 0.75)]
T.append(r"\clearpage\sect{6. Plots}" + "\n")
for f, cap, w in figs:
    T.append(f"\\begin{{figure}}[H]\\centering\\includegraphics[width={w}\\textwidth]{{{os.path.join(FIG, f)}}}\\caption{{{cap}}}\\end{{figure}}\n")

def verb(title, path):
    title = title.replace('_', chr(92) + '_')
    T.append(f"\\subsection*{{{title}}}\\noindent\\texttt{{{os.path.relpath(path, ROOT).replace('_', chr(92)+'_')}}}\n\\begin{{Verbatim}}\n" + open(path).read().rstrip() + "\n\\end{Verbatim}\n")
T.append(r"\clearpage\sect{7. Console output (real program output)}" + "\n")
L = os.path.join(ROOT, 'logs'); C = os.path.join(R, 'console')
verb('Training log: proposed model (fact, r = 4, seed 42)', os.path.join(L, 'model_fact_rank_4_seed_42_save.log'))
verb('Training log: existing model (dense, seed 42)', os.path.join(L, 'model_dense_seed_42_save.log'))
verb('Training log: real-valued control (seed 42)', os.path.join(L, 'model_real_seed_42_save.log'))
verb('Demo: python src/predict.py (proposed model)', os.path.join(C, 'demo_predict_proposed.txt'))
verb('Demo: python src/predict.py --model dense (existing model)', os.path.join(C, 'demo_predict_existing.txt'))
verb('python src/explain_compression.py', os.path.join(C, 'explain_compression.txt'))
T.append(f"""\\sect{{8. Summary of findings}}
\\begin{{itemize}}
\\item The proposed factorized complex embedding stores \\textbf{{{F['emb']:,}}} instead of {D['emb']:,} embedding values
 (\\textbf{{{D['emb']/F['emb']:.0f}$\\times$ fewer}}), uses {100*(1-PR['fact']['peak_train_mb']/PR['dense']['peak_train_mb']):.1f}\\% less peak training memory
 and trains {100*(1-F['tpe'][0]/D['tpe'][0]):.1f}\\% faster per epoch than the existing dense complex model.
\\item Its test accuracy is {F['acc'][0]:.2f}\\% against {D['acc'][0]:.2f}\\% for the dense model ({sg(F['acc'][0]-D['acc'][0])} points);
 most extra errors are negative sentences predicted as positive (false positive rate {F['fpr'][0]:.2f}\\% vs {D['fpr'][0]:.2f}\\%).
\\item It is the most stable model across seeds (accuracy s.d.\\ {F['acc'][1]:.2f} vs {D['acc'][1]:.2f} and {Rl['acc'][1]:.2f}).
\\item A higher rank narrows the gap: $r=8$ reaches {100*get('fact',8,42)[0]['test']['acc']:.2f}\\% with {get('fact',8,42)[0]['emb_params']:,} embedding values.
\\item The real-valued control ({Rl['acc'][0]:.2f}\\%) performs comparably to the dense complex model, so the complex representation alone gave no accuracy gain on SST-2.
\\end{{itemize}}
\\vspace{{6pt}}\\noindent\\small All values were produced by \\texttt{{src/qwe.py}} (12 training runs) and \\texttt{{src/profile\\_res.py}}, and collected by this script from \\texttt{{results/*.json}}.
\\end{{document}}
""")
tex = os.path.join(R, 'Group14_Quantum_Inspired_Word_Embeddings_Results.tex')
open(tex, 'w', encoding='utf8').write(''.join(T))
for _ in range(2):
    subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error', '-output-directory', R, tex], check=True, stdout=subprocess.DEVNULL)
for ext in ('aux', 'log', 'out'):
    try: os.remove(os.path.join(R, f'Group14_Quantum_Inspired_Word_Embeddings_Results.{ext}'))
    except OSError: pass
print('Wrote', os.path.join(R, 'Group14_Quantum_Inspired_Word_Embeddings_Results.pdf'))
