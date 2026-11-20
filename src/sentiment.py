"""Loads a trained model from checkpoints/ and classifies sentences. Used by predict.py and app.py."""
import os, sys, re, pickle, runpy

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
TAGS = {'fact': 'fact_r4_s42', 'dense': 'dense_s42', 'real': 'real_s42'}
NAMES = {'fact': 'Factorized complex (proposed)', 'dense': 'Dense complex (existing)', 'real': 'Real-valued control'}


def tokenize(text):
    """Match SST-2 style: lowercase, split punctuation and clitics ("isn't" -> "is n't")."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"n't\b", " n't", t)
    t = re.sub(r"'(s|re|ve|ll|d|m)\b", r" '\1", t)
    t = re.sub(r"([.,!?;:()\"])", r" \1 ", t)
    return t.split()


class SentimentModel:
    def __init__(self, model='fact'):
        path = os.path.join(ROOT, 'checkpoints', TAGS[model] + '.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'{path} not found. Train it first:  python src/qwe.py --model {model}'
                + (' --rank 4' if model == 'fact' else '') + ' --seed 42 --save')
        saved = pickle.load(open(path, 'rb'))
        a = saved['args']
        old_argv = sys.argv
        sys.argv = ['qwe.py', '--model', a['model'], '--rank', str(a['rank']), '--seed', str(a['seed']), '--tau', str(a['tau'])]
        os.environ['QWE_NO_TRAIN'] = '1'
        try:   # load the exact model definition used in training, then plug in the trained weights
            g = runpy.run_path(os.path.join(HERE, 'qwe.py'), run_name='__main__')
        finally:
            sys.argv = old_argv
        g['P'].clear(); g['P'].update(saved['params'])
        self.g, self.forward, self.np = g, g['forward'], g['onp']
        self.MAXL = saved['MAXL']; self.stoi = {w: i for i, w in enumerate(saved['itos'])}
        self.name = NAMES[model]; self.tag = TAGS[model]
        self.n_params = int(sum(v.size for v in g['flat'](saved['params']).values()))

    def classify(self, text):
        """Returns (label, p_negative, p_positive, unknown_words)."""
        np = self.np
        toks = tokenize(text)[:self.MAXL] or ['<unk>']
        ids = np.zeros((1, self.MAXL), dtype=int); m = np.zeros((1, self.MAXL))
        ids[0, :len(toks)] = [self.stoi.get(w, 1) for w in toks]; m[0, :len(toks)] = 1
        z = np.array(self.forward(self.g['P'], ids, m))[0]
        pr = np.exp(z - z.max()); pr /= pr.sum()
        unknown = [w for w in toks if w not in self.stoi]
        return ('POSITIVE' if pr[1] > pr[0] else 'NEGATIVE'), float(pr[0]), float(pr[1]), unknown
