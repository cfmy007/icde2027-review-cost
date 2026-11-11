"""Bootstrap paired total-only versus verified-class-matched Wiki interventions."""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).parent


def metrics(pred, truth):
    y = truth[None, None, :]
    pp = np.sum(pred == 1, axis=2)
    gp = np.sum(y == 1, axis=2)
    tp = np.sum((pred == 1) & (y == 1), axis=2)
    total = pred.shape[2]
    tn = total - pp - gp + tp
    denp, denn = pp + gp, 2 * total - pp - gp
    f1p = np.divide(2 * tp, denp, out=np.zeros_like(tp, dtype=float), where=denp > 0)
    f1n = np.divide(2 * tn, denn, out=np.zeros_like(tn, dtype=float), where=denn > 0)
    return (f1p + f1n) / 2, (tp + tn) / total


def run(learner):
    mats, ids, gold, counts = [], [], [], []
    for split in range(10):
        values, c = [], []
        for rep in range(10):
            with np.load(R / 'equal_cost_class_match' / learner / (str(415500 + 10 * rep + split) + '.npz')) as d:
                values.append(d['predictions'])
                c.append([len(d['total_source']), len(np.intersect1d(d['total_source'], d['total_validation']))])
                if rep == 0:
                    ids.append(d['test_ids'])
                    gold.append(d['test_reference'])
        mats.append(np.stack(values))
        counts.append(c)
    mats = np.stack(mats)
    point_f1, point_acc = zip(*(metrics(mats[s], gold[s]) for s in range(10)))
    pf, pa = np.stack(point_f1), np.stack(point_acc)
    unique = np.unique(np.concatenate(ids))
    loc = [np.searchsorted(unique, v) for v in ids]
    rng = np.random.default_rng(429000 + (learner == 'mlp'))
    draws = {'f1': [], 'accuracy': []}
    for _ in range(100):
        size = 20
        weights = rng.poisson(1, (size, len(unique))).astype(np.float32)
        estimates, accuracies = [], []
        for split in range(10):
            w = weights[:, loc[split]]
            flat = mats[split].reshape(-1, len(loc[split]))
            y = gold[split] == 1
            pos = (flat == 1).astype(np.float32)
            total = np.maximum(w.sum(1)[:, None], 1)
            pp = w @ pos.T
            gp = (w @ y.astype(np.float32))[:, None]
            tp = (w * y[None, :]) @ pos.T
            tn = total - pp - gp + tp
            denp, denn = pp + gp, 2 * total - pp - gp
            f1 = (np.divide(2 * tp, denp, out=np.zeros_like(tp), where=denp > 0) + np.divide(2 * tn, denn, out=np.zeros_like(tn), where=denn > 0)) / 2
            estimates.append(f1.reshape(size, 10, 4))
            accuracies.append(((tp + tn) / total).reshape(size, 10, 4))
        groups = rng.integers(0, 10, (size, 10))
        seeds = rng.integers(0, 10, (size, 10, 10))
        ix = np.arange(size)[:, None, None]
        for key, estimates_ in [('f1', estimates), ('accuracy', accuracies)]:
            m = np.stack(estimates_, axis=1)[ix, groups[:, :, None], seeds, :].mean(2)
            total_delta = (m[:, :, 1] - m[:, :, 0]).mean(1) * 100
            class_delta = (m[:, :, 3] - m[:, :, 2]).mean(1) * 100
            draws[key].extend(np.stack((total_delta, class_delta, total_delta - class_delta), axis=1))
    out = {'learner': learner, 'matched_corrections_mean': float(np.mean(np.asarray(counts)[:, :, 0])),
           'common_corrections_mean': float(np.mean(np.asarray(counts)[:, :, 1])),
           'scope': 'Post-result offline intervention, same Wikipedia corpus and same paired training seeds. The class match changes individual corrections too.'}
    for key, p in [('f1', pf), ('accuracy', pa)]:
        total = float((p[:, :, 1] - p[:, :, 0]).mean() * 100)
        cls = float((p[:, :, 3] - p[:, :, 2]).mean() * 100)
        b = np.asarray(draws[key])
        out[key] = {'total_only_delta_pp': total, 'total_only_ci95': np.percentile(b[:, 0], [2.5, 97.5]).tolist(),
                    'class_matched_delta_pp': cls, 'class_matched_ci95': np.percentile(b[:, 1], [2.5, 97.5]).tolist(),
                    'interaction_pp': total - cls, 'interaction_ci95': np.percentile(b[:, 2], [2.5, 97.5]).tolist()}
    return out


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('learner', choices=('linear', 'mlp'))
    a = p.parse_args()
    out = run(a.learner)
    (R / ('equal_cost_class_match_' + a.learner + '_summary.json')).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))
