import argparse
import csv
import gzip
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent
NAMES = ['word-head', 'char-head', 'svd-margin', 'svd-knn', 'word-ridge', 'char-ridge']

def values(fit, other, y):
    models = []
    for mode, grams, limit in [('word', (1, 2), 18000), ('char', (3, 5), 24000)]:
        tf = TfidfVectorizer(analyzer=mode, ngram_range=grams, min_df=2, max_features=limit, sublinear_tf=True, dtype=np.float32)
        a, b = tf.fit_transform(fit), tf.transform(other)
        model = LogisticRegression(max_iter=500, random_state=26112499).fit(a, y)
        models.append((a, b, model.predict_proba(b)))
    a, b, word = models[0]
    char = models[1][2]
    svd = TruncatedSVD(min(64, a.shape[1]-1, a.shape[0]-1), random_state=26112499).fit(a)
    x, z = normalize(svd.transform(a)), normalize(svd.transform(b))
    centers = normalize(np.stack([x[y == c].mean(0) for c in (0, 1)]))
    sim = z @ centers.T
    ix = NearestNeighbors(n_neighbors=min(10, len(y)), algorithm='brute').fit(x).kneighbors(z, return_distance=False)
    nn = np.stack([(y[ix] != c).sum(1) for c in (0, 1)], axis=1).astype(float)
    ridge = []
    for a, b, _ in models:
        v = RidgeClassifier(alpha=1).fit(a, y).decision_function(b)
        ridge.append(-np.log(softmax(np.stack([-v, v], axis=1), axis=1)))
    return np.stack([-np.log(np.clip(word, 1e-12, 1)), -np.log(np.clip(char, 1e-12, 1)), sim.max(1, keepdims=True)-sim, nn, *ridge]), char

def ranks(mats, labels, tie):
    return np.stack([tie[np.argsort(-v[np.arange(len(labels)), labels][tie], kind='stable')] for v in mats])

def run(kind, out):
    with gzip.open(ROOT/'inputs'/f'{kind}.jsonl.gz', 'rt', encoding='utf8') as f:
        rows = [json.loads(line) for line in f]
    fit, src, tgt = [[r for r in rows if r['role'] == i] for i in range(3)]
    y = np.array([r['reference'] for r in fit], int)
    sy = np.array([r['reference'] for r in src], int)
    ty = np.array([r['reference'] for r in tgt], int)
    claim = np.array([r['claim'] for r in tgt], int)
    mats, prob = values([r['text'] for r in fit], [r['text'] for r in src+tgt], y)
    m = len(src)
    tie = np.argsort([r['tie'] for r in tgt])
    target = ranks(mats[:, m:], claim, tie)
    difficulty = prob[np.arange(m), 1-sy]
    output = []
    for seed in range(26112500, 26112510):
        rng = np.random.default_rng(seed)
        ix = np.concatenate([rng.permutation(np.flatnonzero(sy == c))[:max(1, round(.8*(sy == c).sum()))] for c in (0, 1)])
        z = sy[ix]
        k = round(.2*len(ix))
        t = rng.permutation(len(ix))
        masks = (rng.permutation(len(ix))[:k], t[np.argsort(-difficulty[ix][t], kind='stable')[:k]])
        source_orders = []
        for mask in masks:
            corrupt = z.copy()
            corrupt[mask] = 1-corrupt[mask]
            source_orders.append((corrupt != z)[ranks(mats[:, ix], corrupt, t)])
        for budget in (.05, .10, .20):
            hits = np.stack([s[:, :round(len(ix)*budget)].sum(1) for s in source_orders])
            choices = hits.argmax(1)
            output.append(dict(seed=seed, budget=budget, menu='expanded', source_n=len(ix), random=int(choices[0]), difficulty=int(choices[1]), random_source=hits[0].tolist(), difficulty_source=hits[1].tolist()))
    out.mkdir(parents=True, exist_ok=True)
    with (out/'paired.csv').open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(output[0])); w.writeheader(); w.writerows(output)
    np.savez_compressed(out/'target.npz', errors=claim != ty, rank=target, names=np.array(NAMES), ids=np.arange(len(tgt)), group=np.array([r['group'] for r in tgt]))
    print(kind, len(fit), len(src), len(tgt), int((claim != ty).sum()), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('datasets', nargs='+')
    p.add_argument('--output', type=Path, default=ROOT/'generated')
    a = p.parse_args()
    for kind in a.datasets:
        run(kind, a.output/kind)
