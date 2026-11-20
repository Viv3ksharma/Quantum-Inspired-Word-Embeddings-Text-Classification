"""Small desktop demo (Tkinter, comes with Python): type a sentence, see the predicted sentiment
from the proposed (factorized) model and the existing (dense) model side by side.

Usage:  python src/app.py
"""
import os, sys
import tkinter as tk
from tkinter import ttk
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sentiment import SentimentModel

models = {}
for key in ['fact', 'dense']:
    try:
        models[key] = SentimentModel(key)
    except FileNotFoundError as e:
        print(e)
if not models:
    sys.exit('No trained models found in checkpoints/. Run run_all (or qwe.py --save) first.')

root = tk.Tk(); root.title('Quantum-Inspired Sentiment Classifier - Group 14')
root.geometry('760x420')
MAROON = '#8C1616'
tk.Label(root, text='Quantum-Inspired Word Embeddings for Text Classification', font=('Arial', 14, 'bold'), fg=MAROON).pack(pady=(14, 2))
tk.Label(root, text='Saurabh Tripathi & Vivek Sharma  |  K J Somaiya School of Engineering', font=('Arial', 10)).pack()
tk.Label(root, text='Enter a movie-review sentence:', font=('Arial', 11)).pack(anchor='w', padx=20, pady=(16, 4))
entry = tk.Entry(root, font=('Arial', 12)); entry.pack(fill='x', padx=20)
entry.insert(0, 'the movie was not good at all')

frame = tk.Frame(root); frame.pack(fill='both', expand=True, padx=20, pady=12)
outputs = {}
for col, key in enumerate(models):
    m = models[key]
    box = tk.LabelFrame(frame, text=f'{m.name}  ({m.n_params:,} params)', font=('Arial', 10, 'bold'), padx=10, pady=10)
    box.grid(row=0, column=col, sticky='nsew', padx=6); frame.columnconfigure(col, weight=1)
    lab = tk.Label(box, text='-', font=('Arial', 20, 'bold')); lab.pack()
    bar = ttk.Progressbar(box, length=280, maximum=100); bar.pack(pady=6)
    det = tk.Label(box, text='', font=('Arial', 10)); det.pack()
    outputs[key] = (lab, bar, det)

def run(*_):
    s = entry.get().strip()
    if not s:
        return
    for key, m in models.items():
        lab, pn, pp, unk = m.classify(s)
        w, bar, det = outputs[key]
        w.config(text=('😊 ' if lab == 'POSITIVE' else '😠 ') + lab, fg='#1a7f37' if lab == 'POSITIVE' else '#b42318')
        bar['value'] = 100 * pp
        det.config(text=f'positive {100*pp:.1f}%  |  negative {100*pn:.1f}%' + (f'\nunknown words: {", ".join(unk)}' if unk else ''))

tk.Button(root, text='Classify', font=('Arial', 12, 'bold'), bg=MAROON, fg='white', command=run).pack(pady=(0, 14))
entry.bind('<Return>', run)
run()
root.mainloop()
