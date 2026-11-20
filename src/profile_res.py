"""Resource-utilization profiling (Table 8 of the paper): peak memory and CPU usage for one
training epoch and for test-set inference, for one model (seed 42).

Usage:  python src/profile_res.py fact      (or dense / real)
"""
import os, sys, time, json, tracemalloc, runpy

model = sys.argv[1] if len(sys.argv) > 1 else 'fact'
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.argv = ['qwe.py', '--model', model, '--rank', '4', '--seed', '42']
os.environ['QWE_NO_TRAIN'] = '1'          # load data + model definition only
g = runpy.run_path(os.path.join(HERE, 'qwe.py'), run_name='__main__')
onp, P, vg, predict = g['onp'], g['P'], g['vg'], g['predict']
Itr, Mtr, Ytr, Ite, Mte = g['Itr'], g['Mtr'], g['Ytr'], g['Ite'], g['Mte']

def run_epoch(P):
    perm = onp.random.RandomState(43).permutation(len(Itr))
    for s in range(0, len(perm), 32):
        b = perm[s:s + 32]
        l, gr = vg(P, Itr[b], Mtr[b], Ytr[b])
        P = {k: (v - 1e-3 * gr[k] if not isinstance(v, dict) else {kk: vv - 1e-3 * gr[k][kk] for kk, vv in v.items()})
             for k, v in P.items()}
    return P

tracemalloc.start(); c0 = time.process_time(); w0 = time.time()
P = run_epoch(P)
c1 = time.process_time(); w1 = time.time(); _, peak_train = tracemalloc.get_traced_memory(); tracemalloc.stop()
tracemalloc.start(); c2 = time.process_time(); w2 = time.time()
predict(P, Ite, Mte)
c3 = time.process_time(); w3 = time.time(); _, peak_inf = tracemalloc.get_traced_memory(); tracemalloc.stop()
try:                                   # peak process memory (Linux/macOS); not available on Windows
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
except ImportError:
    rss = float('nan')

res = dict(model=model, peak_train_mb=peak_train / 2**20, peak_inf_mb=peak_inf / 2**20, rss_mb=rss,
           cpu_train_s=c1 - c0, wall_train_s=w1 - w0, cpu_util_train=100 * (c1 - c0) / (w1 - w0),
           cpu_inf_s=c3 - c2, wall_inf_s=w3 - w2, cpu_util_inf=100 * (c3 - c2) / (w3 - w2))
os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, 'results', f'profile_{model}.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
