"""Demo: classify your own sentences as POSITIVE or NEGATIVE with a trained model.

Usage:
  python src/predict.py                                   # interactive mode (type sentences)
  python src/predict.py "the movie was not good"          # one sentence
  python src/predict.py --model dense "a wonderful film"  # use the dense (existing) model instead
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sentiment import SentimentModel

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='fact', choices=['fact', 'dense', 'real'])
ap.add_argument('sentence', nargs='*')
args = ap.parse_args()

try:
    sm = SentimentModel(args.model)
except FileNotFoundError as e:
    sys.exit(str(e))
print(f'Loaded model: {sm.name}  ({sm.n_params:,} parameters)')

def show(s):
    lab, pn, pp, unk = sm.classify(s)
    print(f'  {lab:8s}  (negative {100*pn:5.1f}% | positive {100*pp:5.1f}%)' + (f'   unknown words: {unk}' if unk else ''))

if args.sentence:
    s = ' '.join(args.sentence); print(f'"{s}"'); show(s)
else:
    print('Type a movie-review sentence (press Enter on an empty line to quit).')
    while True:
        try:
            s = input('> ').strip()
        except EOFError:
            break
        if not s:
            break
        show(s)
