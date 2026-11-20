"""Quantum-inspired (factorized complex) word embeddings for SST-2 sentiment classification.

Group 14 - Saurabh Tripathi & Vivek Sharma
Pure NumPy + HIPS autograd implementation (runs on any CPU, no GPU needed).

Models (choose with --model):
  real        : real-valued embedding + LSTM + mean pooling + linear softmax (real-valued control)
  dense       : dense complex embedding (E_R, E_I) + dual LSTM + density-mixture overlap head (EXISTING method)
  fact        : word2ketXS-factorized complex embedding + same encoder/head (PROPOSED method)
  fact_noimag : factorized, imaginary channel removed (ablation)

Example:
  python src/qwe.py --model fact --rank 4 --seed 42 --save
"""
import os, time, json, math, collections, argparse, pickle
import numpy as onp
import autograd.numpy as np
from autograd import value_and_grad

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser()
p.add_argument('--model', default='fact', choices=['real', 'dense', 'fact', 'fact_noimag'])
p.add_argument('--seed', type=int, default=42)
p.add_argument('--rank', type=int, default=4)
p.add_argument('--d', type=int, default=64)
p.add_argument('--h', type=int, default=64)
p.add_argument('--epochs', type=int, default=20)
p.add_argument('--patience', type=int, default=4)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--tau', type=float, default=0.1)
p.add_argument('--bs', type=int, default=32)
p.add_argument('--data', default=os.path.join(ROOT, 'data', 'SST2'))
p.add_argument('--out', default=os.path.join(ROOT, 'results'))
p.add_argument('--save', action='store_true', help='save the trained model to checkpoints/ (used by predict.py)')
a = p.parse_args()
rng = onp.random.RandomState(a.seed)

# ---------------- data ----------------
def load(f):
    X, Y = [], []
    for line in open(f, encoding='utf8'):
        line = line.rstrip('\n')
        if not line.strip():
            continue
        t, y = line.rsplit('\t', 1)
        t = ' '.join(t.split())
        if not t:
            continue
        X.append(t.split()); Y.append(int(y))
    return X, onp.array(Y)
Xtr, Ytr = load(os.path.join(a.data, 'train.tsv'))
Xva, Yva = load(os.path.join(a.data, 'dev.tsv'))
Xte, Yte = load(os.path.join(a.data, 'test.tsv'))
cnt = collections.Counter(w for s in Xtr for w in s)
itos = ['<pad>', '<unk>'] + sorted(cnt)
stoi = {w: i for i, w in enumerate(itos)}
V = len(itos)
lens = onp.array([len(s) for s in Xtr])
MAXL = int(onp.percentile(lens, 99))
def enc(X):
    ids = onp.zeros((len(X), MAXL), dtype=int); m = onp.zeros((len(X), MAXL))
    for i, s in enumerate(X):
        s = s[:MAXL]
        ids[i, :len(s)] = [stoi.get(w, 1) for w in s]; m[i, :len(s)] = 1
    return ids, m
Itr, Mtr = enc(Xtr); Iva, Mva = enc(Xva); Ite, Mte = enc(Xte)
trunc_rate = float((lens > MAXL).mean())
oov_test = float(onp.mean([stoi.get(w, 1) == 1 for s in Xte for w in s]))

d, h, C, r = a.d, a.h, 2, a.rank
d1 = int(round(math.sqrt(d))); d2 = d // d1
v1 = int(math.ceil(math.sqrt(V))); v2 = int(math.ceil(V / v1))

# ---------------- params ----------------
def lstm_init(din, hh):
    s = 1 / math.sqrt(hh)
    b = onp.zeros(4 * hh); b[hh:2 * hh] = 1.0  # forget-gate bias 1
    return {'Wx': rng.uniform(-s, s, (din, 4 * hh)), 'Wh': rng.uniform(-s, s, (hh, 4 * hh)), 'b': b}
P = {}
M = a.model
if M == 'real':
    P['E'] = rng.normal(0, 0.1, (V, d)); P['lstm'] = lstm_init(d, h)
    P['Wo'] = rng.normal(0, 0.1, (h, C)); P['bo'] = onp.zeros(C)
else:
    chans = ['R'] if M == 'fact_noimag' else ['R', 'I']
    for c in chans:
        if M == 'dense':
            P['E' + c] = rng.normal(0, 0.1, (V, d))
        else:
            sig = (0.01 / r) ** 0.25
            P['F1' + c] = rng.normal(0, sig, (r, v1, d1)); P['F2' + c] = rng.normal(0, sig, (r, v2, d2))
        P['lstm' + c] = lstm_init(d, h)
    P['q'] = rng.normal(0, 1, (C, len(chans) * h)); P['bc'] = onp.zeros(C)

def flat(P, pre=''):
    out = {}
    for k, v in P.items():
        if isinstance(v, dict): out.update(flat(v, pre + k + '.'))
        else: out[pre + k] = v
    return out
FP = flat(P)
n_params = int(sum(v.size for v in FP.values()))
emb_params = int(sum(v.size for k, v in FP.items() if k.startswith(('E', 'F'))))

# ---------------- model ----------------
def lstm(p, X):  # X: (B,L,din) -> (B,L,h)
    B, L, _ = X.shape
    XW = np.dot(X, p['Wx']) + p['b']
    hs = []; ht = np.zeros((B, h)); ct = np.zeros((B, h))
    for t in range(L):
        z = XW[:, t, :] + np.dot(ht, p['Wh'])
        i = 1 / (1 + np.exp(-z[:, :h])); f = 1 / (1 + np.exp(-z[:, h:2 * h]))
        o = 1 / (1 + np.exp(-z[:, 2 * h:3 * h])); g = np.tanh(z[:, 3 * h:])
        ct = f * ct + i * g; ht = o * np.tanh(ct); hs.append(ht)
    return np.stack(hs, axis=1)

def embed(P, ids, c):
    """Real (c='R') or imaginary (c='I') embedding rows for token ids."""
    if M == 'dense': return P['E' + c][ids]
    # word2ketXS: row index w -> grid address (i, j); row = sum_k F1[k,i] (x) F2[k,j]
    i, j = ids // v2, ids % v2
    A = P['F1' + c][:, i, :]; Bm = P['F2' + c][:, j, :]       # (r,B,L,d1),(r,B,L,d2)
    rows = np.einsum('kbli,kblj->blij', A, Bm)                 # Kronecker product, summed over rank
    return np.reshape(rows, (ids.shape[0], ids.shape[1], d))

def forward(P, ids, m):
    L = int(m.sum(1).max()); ids = ids[:, :L]; m = m[:, :L]
    mm = m[:, :, None]
    if M == 'real':
        H = lstm(P['lstm'], P['E'][ids] * mm)
        pooled = np.sum(H * mm, 1) / np.sum(m, 1, keepdims=True)
        return np.dot(pooled, P['Wo']) + P['bo']
    chans = ['R'] if M == 'fact_noimag' else ['R', 'I']
    Hs = [lstm(P['lstm' + c], embed(P, ids, c) * mm) for c in chans]          # a_t, b_t
    nrm = np.sqrt(sum(np.sum(H ** 2, 2, keepdims=True) for H in Hs) + 1e-12)
    U = [H / nrm for H in Hs]                                                   # u_t = g_t/||g_t||
    q = P['q']; qn = q / np.sqrt(np.sum(q ** 2, 1, keepdims=True) + 1e-12)      # class vectors v_c
    if len(chans) == 2:
        vR, vI = qn[:, :h], qn[:, h:]; uR, uI = U
        re = np.einsum('blh,ch->blc', uR, vR) + np.einsum('blh,ch->blc', uI, vI)
        im = np.einsum('blh,ch->blc', uI, vR) - np.einsum('blh,ch->blc', uR, vI)
        ov = re ** 2 + im ** 2                                                  # |v_c^dagger u_t|^2
    else:
        ov = np.einsum('blh,ch->blc', U[0], qn) ** 2
    s = np.sum(ov * mm, 1) / np.sum(m, 1, keepdims=True)                        # s_c = Tr(rho v v^dagger)
    return s / a.tau + P['bc']

def loss(P, ids, m, y):
    z = forward(P, ids, m)
    z = z - np.max(z, 1, keepdims=True)
    lp = z - np.log(np.sum(np.exp(z), 1, keepdims=True))
    return -np.mean(lp[np.arange(len(y)), y])
vg = value_and_grad(loss)

def predict(P, I, Mk, bs=256):
    out = []
    for s in range(0, len(I), bs):
        out.append(onp.argmax(forward(P, I[s:s + bs], Mk[s:s + bs]), 1))
    return onp.concatenate(out)

def metrics(y, pr):
    cm = onp.zeros((2, 2), int)
    for t, q in zip(y, pr): cm[t, q] += 1
    P_, R_, F_ = [], [], []
    for c in range(2):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        pc = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); fc = 2 * pc * rc / max(pc + rc, 1e-12)
        P_.append(pc); R_.append(rc); F_.append(fc)
    fpr = cm[0, 1] / max(cm[0].sum(), 1)
    return dict(acc=float(onp.trace(cm) / cm.sum()), prec=float(onp.mean(P_)), rec=float(onp.mean(R_)),
                f1=float(onp.mean(F_)), f1_neg=float(F_[0]), f1_pos=float(F_[1]),
                spec=float(cm[0, 0] / max(cm[0].sum(), 1)), fpr=float(fpr), err=float(1 - onp.trace(cm) / cm.sum()),
                cm=cm.tolist())

if __name__ == '__main__' and os.environ.get('QWE_NO_TRAIN') != '1':
    # ---------------- training (Adam) ----------------
    def tree_map(f, *ts):
        if isinstance(ts[0], dict): return {k: tree_map(f, *[t[k] for t in ts]) for k in ts[0]}
        return f(*ts)
    mS = tree_map(onp.zeros_like, P); vS = tree_map(onp.zeros_like, P)
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0; best = -1; best_P = None; bad = 0; hist = []; t_train = 0.0
    order_rng = onp.random.RandomState(a.seed + 1)
    print(f'model={M} seed={a.seed} rank={r} | vocab={V} params={n_params:,} embedding params={emb_params:,}', flush=True)
    for ep in range(a.epochs):
        t0 = time.time(); tot = 0
        # length-bucketed batches (sort by noisy length, then shuffle batch order)
        perm = onp.argsort(Mtr.sum(1) + order_rng.uniform(0, 3, len(Itr)))
        batches = [perm[s:s + a.bs] for s in range(0, len(perm), a.bs)]
        order_rng.shuffle(batches)
        for b in batches:
            l, g = vg(P, Itr[b], Mtr[b], Ytr[b]); tot += l * len(b)
            gn = math.sqrt(sum(float(onp.sum(x ** 2)) for x in flat(g).values()))
            sc = min(1.0, 5.0 / (gn + 1e-12)); step += 1
            def upd(p_, g_, m_, v_):
                g_ = g_ * sc
                m_[...] = b1 * m_ + (1 - b1) * g_; v_[...] = b2 * v_ + (1 - b2) * g_ ** 2
                mh = m_ / (1 - b1 ** step); vh = v_ / (1 - b2 ** step)
                return p_ - a.lr * mh / (onp.sqrt(vh) + eps)
            P = tree_map(upd, P, g, mS, vS)
        et = time.time() - t0; t_train += et
        va = metrics(Yva, predict(P, Iva, Mva))
        hist.append(dict(epoch=ep + 1, loss=float(tot / len(perm)), val_acc=va['acc'], val_f1=va['f1'], time=et))
        print(json.dumps(hist[-1]), flush=True)
        if va['f1'] > best: best = va['f1']; best_P = tree_map(onp.copy, P); bad = 0
        else:
            bad += 1
            if bad >= a.patience: break
    P = best_P
    t0 = time.time(); pr = predict(P, Ite, Mte); lat = (time.time() - t0) / len(Ite) * 1000
    res = dict(model=M, seed=a.seed, rank=r if M.startswith('fact') else None, V=V, MAXL=MAXL, trunc_rate=trunc_rate,
               oov_test=oov_test, n_params=n_params, emb_params=emb_params, emb_kb=emb_params * 4 / 1024,
               train_time_s=t_train, epochs_run=len(hist), time_per_epoch=t_train / len(hist), latency_ms=lat,
               best_val_f1=best, val=metrics(Yva, predict(P, Iva, Mva)), test=metrics(Yte, pr), hist=hist,
               v1=v1, v2=v2, d1=d1, d2=d2)
    tag = f"{M}_r{r}_s{a.seed}" if M.startswith('fact') else f"{M}_s{a.seed}"
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"{tag}.json"), 'w'), indent=1)
    t = res['test']
    print(f"TEST  accuracy={100*t['acc']:.2f}%  precision={100*t['prec']:.2f}%  recall={100*t['rec']:.2f}%  F1={100*t['f1']:.2f}%")
    if a.save:
        os.makedirs(os.path.join(ROOT, 'checkpoints'), exist_ok=True)
        with open(os.path.join(ROOT, 'checkpoints', f'{tag}.pkl'), 'wb') as fh:
            pickle.dump(dict(params=P, itos=itos, args=vars(a), V=V, MAXL=MAXL, v1=v1, v2=v2, d1=d1, d2=d2), fh)
        print('saved model to', os.path.join('checkpoints', f'{tag}.pkl'))
