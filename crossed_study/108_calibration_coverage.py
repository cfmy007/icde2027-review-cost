"""Recompute the archived 200-split coverage experiment without phantom reviews."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_features import HERE

SETTINGS = {
    "CIFAR-100N 40.2%": "cifar100n/probe_CNN64.npz",
    "CIFAR-10N worse 40.2%": "cifar10n/probe_worse_label_CNN64.npz",
    "CIFAR-10N aggre 9.0%": "cifar10n/probe_aggre_label_CNN64.npz",
    **{f"CIFAR-10N random{i} {rate}%": f"noise_ladder/probe_random_label{i}.npz"
       for i, rate in enumerate(('17.2', '18.1', '17.6'), 1)},
}
TARGETS = (.50, .70, .85, .95)
MARGINS = (0., .025, .05, .075, .10)
BUDGETS = np.round(np.arange(.01, .501, .005), 3)
N_SPLITS = 200


def precision_curve(err, score):
    hits = np.cumsum(err[np.argsort(-score)])
    for b in BUDGETS:
        k = int(round(b * len(err)))
        if k:
            yield b, k, hits[k-1] / k


def pick_budget(curve, threshold, min_n=20):
    best = 0.
    for b, k, p in curve:
        if k < min_n:
            continue
        if p < threshold:
            break
        best = float(b)
    return best


def evaluate(err, score, b):
    k = int(round(b * len(err)))
    hits = int(err[np.argsort(-score)[:k]].sum()) if k else 0
    return dict(reviewed=k, gain_count=hits, precision=hits/k if k else None,
                gain_pp=100*hits/len(err), executed=bool(k))


def run(base, out):
    records = []
    for alpha in (.50, .10):
        # Restart stream as in the two archived invocations of script 108.
        rng = np.random.default_rng(0)
        for name, rel in SETTINGS.items():
            with np.load(base / 'results' / rel) as z:
                err, score = z['err'].astype(bool), z['conf']
            n = len(err)
            for target in TARGETS:
                for rep in range(N_SPLITS):
                    perm = rng.permutation(n)
                    na = max(20, int(round(alpha*n)))
                    audit, te = perm[:na], perm[na:]
                    curve = tuple(precision_curve(err[audit], score[audit]))
                    for margin in MARGINS:
                        b = pick_budget(curve, target+margin)
                        row = evaluate(err[te], score[te], b)
                        p = row['precision']
                        records.append(dict(dataset=name, alpha=alpha, target=target, repeat=rep, margin=margin,
                                            n=n, audit=na, audit_errors=int(err[audit].sum()),
                                            budget=b, total_cost=na+row['reviewed'],
                                            meets=p is not None and p >= target, **row))
    frame = pd.DataFrame(records)
    out.mkdir(exist_ok=True, parents=True)
    frame.to_csv(out/'legacy_coverage_rows.csv', index=False)
    summary = []
    for key, z in frame.groupby(['dataset', 'alpha', 'target', 'margin']):
        z1 = z[z.executed]
        summary.append(dict(zip(('dataset', 'alpha', 'target', 'margin'), key),
                            execution_rate=z.executed.mean(), precision_cond=z1.precision.mean(),
                            met_unconditional=z.meets.mean(), met_conditional=z1.meets.mean(),
                            gain_pp_probe=100*(z.gain_count/z.n).mean(), mean_reviewed=z.reviewed.mean(),
                            mean_total_cost=z.total_cost.mean(), total_cost_pct=100*(z.total_cost/z.n).mean()))
    pd.DataFrame(summary).to_csv(out/'legacy_coverage_summary.csv', index=False)


if __name__ == '__main__':
    assert evaluate(np.ones(10), np.arange(10), 0.) == {
        'reviewed': 0, 'gain_count': 0, 'precision': None, 'gain_pp': 0., 'executed': False}
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=HERE/'results')
    args = parser.parse_args()
    run(args.base, args.out)
