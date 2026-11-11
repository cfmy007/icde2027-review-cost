"""Paired partition, seed and shared-test-identity bootstrap for paid selection."""
import argparse, csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent


def metrics(pred, truth):
    if pred.dtype == bool:
        return pred.mean(axis=-1)
    y = truth[None, None, :]
    positive = pred == 1
    gt = y == 1
    tp = np.sum(positive & gt, axis=-1)
    pp = np.sum(positive, axis=-1)
    gp = np.sum(gt, axis=-1)
    total = pred.shape[-1]
    tn = total - pp - gp + tp
    f1p = np.divide(2 * tp, pp + gp, out=np.zeros_like(tp, dtype=float), where=(pp + gp) > 0)
    f1n = np.divide(2 * tn, 2 * total - pp - gp, out=np.zeros_like(tp, dtype=float), where=(2 * total - pp - gp) > 0)
    return (f1p + f1n) / 2


def run(dataset, learner, draws=2000):
    folder = ROOT / 'equal_cost_selection' / dataset / learner
    base = 419500 if dataset == 'c100' else 415500
    matrices, repaired, testids, truth, records = [], [], [], [], []
    names = methods = None
    for split in range(10):
        p, h = [], []
        for rep in range(10):
            with np.load(folder / (str(base + 10 * rep + split) + '.npz')) as d:
                if names is None:
                    names, methods = d['policies'].tolist(), d['methods'].tolist()
                assert d['policies'].tolist() == names
                v = np.unpackbits(d['predictions'], axis=1, count=int(d['n_test'])).astype(bool) if dataset == 'c100' else d['predictions']
                p.append(v)
                h.append(d['repairs'])
                if rep == 0:
                    testids.append(d['test_ids'])
                    truth.append(d['test_reference'])
                else:
                    assert np.array_equal(testids[-1], d['test_ids']) and np.array_equal(truth[-1], d['test_reference'])
                records.append({'dataset': dataset, 'learner': learner, 'split': split, 'repeat': rep, 'seed': int(d['seed']),
                                'audit_selected': str(d['audit_selected']), 'validation_selected': str(d['validation_selected']),
                                'union_n': int(d['union_n'])})
        matrices.append(np.stack(p))
        repaired.append(np.stack(h))
    matrices = np.stack(matrices)
    repaired = np.stack(repaired)
    point = np.stack([metrics(matrices[s], truth[s]) for s in range(10)])
    oracle_indices = np.array([np.argmax(point[s].mean(0)[[i for i, n in enumerate(names) if n.startswith('oracle/')]]) for s in range(10)])
    oracle_candidates = np.array([i for i, n in enumerate(names) if n.startswith('oracle/')])
    oracle_indices = oracle_candidates[oracle_indices]
    deploy = [i for i, n in enumerate(names) if not n.startswith('oracle/')]
    source = 'source_self' if dataset == 'c100' else 'source'
    pairs = [('yield_audit', source), ('validation_select', source), ('validation_select', 'yield_audit'),
             ('random', source), (source, 'no_review'), ('validation_select', 'random'),
             ('yield_audit', 'random'), ('validation_select', 'no_review'), ('random', 'no_review')]
    if dataset == 'c100':
        pairs.extend([('yield_audit', 'source_external'), ('validation_select', 'source_external'), ('source_external', 'source_self')])
    labels = [a + '_minus_' + b for a, b in pairs] + ['oracle_minus_' + names[i] for i in deploy if names[i] != 'no_review']
    boot = {k: [] for k in labels}
    boot_repairs = {a + '_minus_' + b: [] for a, b in pairs}
    boot_accuracy = {a + '_minus_' + b: [] for a, b in pairs} if dataset == 'wiki' else None
    unique = np.unique(np.concatenate(testids))
    loc = [np.searchsorted(unique, a) for a in testids]
    rng = np.random.default_rng(427000 + (dataset == 'wiki') * 20 + (learner == 'mlp'))
    for start in range(0, draws, 20):
        size = min(20, draws - start)
        weights = rng.poisson(1, (size, len(unique))).astype(np.float32)
        estimates = []
        accuracy_estimates = []
        for split in range(10):
            w = weights[:, loc[split]]
            flat = matrices[split].reshape(-1, len(loc[split]))
            total = np.maximum(w.sum(1)[:, None], 1)
            if dataset == 'c100':
                estimate = (w @ flat.astype(np.float32).T) / total
            else:
                y = truth[split] == 1
                pos = (flat == 1).astype(np.float32)
                pp = w @ pos.T
                gp = (w @ y.astype(np.float32))[:, None]
                tp = (w * y[None, :]) @ pos.T
                tn = total - pp - gp + tp
                denp, denn = pp + gp, 2 * total - pp - gp
                estimate = (np.divide(2 * tp, denp, out=np.zeros_like(tp), where=denp > 0) +
                            np.divide(2 * tn, denn, out=np.zeros_like(tn), where=denn > 0)) / 2
                accuracy_estimates.append(((tp + tn) / total).reshape(size, 10, len(names)))
            estimates.append(estimate.reshape(size, 10, len(names)))
        a = np.stack(estimates, axis=1)
        groups = rng.integers(0, 10, (size, 10))
        seeds = rng.integers(0, 10, (size, 10, 10))
        ix = np.arange(size)[:, None, None]
        m = a[ix, groups[:, :, None], seeds, :].mean(axis=2)
        if dataset == 'wiki':
            am = np.stack(accuracy_estimates, axis=1)[ix, groups[:, :, None], seeds, :].mean(axis=2)
        repairs = repaired[groups[:, :, None], seeds, :].mean(axis=2)
        for left, right in pairs:
            key = left + '_minus_' + right
            boot[key].extend(((m[:, :, names.index(left)] - m[:, :, names.index(right)]).mean(axis=1) * 100).tolist())
            boot_repairs[key].extend((repairs[:, :, names.index(left)] - repairs[:, :, names.index(right)]).mean(axis=1).tolist())
            if dataset == 'wiki':
                boot_accuracy[key].extend(((am[:, :, names.index(left)] - am[:, :, names.index(right)]).mean(axis=1) * 100).tolist())
        b = np.arange(size)[:, None]
        s = np.arange(10)[None, :]
        oracle = m[b, s, oracle_indices[groups]]
        for i in deploy:
            if names[i] != 'no_review':
                boot['oracle_minus_' + names[i]].extend(((oracle - m[:, :, i]).mean(axis=1) * 100).tolist())
    out = {'dataset': dataset, 'learner': learner, 'metric': 'accuracy' if dataset == 'c100' else 'macro-F1',
           'budget': 1000 if dataset == 'c100' else 1200, 'candidate_count': len(methods),
           'mean_union': float(np.mean([r['union_n'] for r in records])), 'methods': methods,
           'policy': {names[i]: {'quality_percent': float(point[:, :, i].mean() * 100),
                                  'mean_repairs': float(repaired[:, :, i].mean())} for i in deploy},
           'contrasts': {}, 'hindsight_oracle': {},
           'oracle_note': 'Oracle chosen using the same held-out test panel after all candidates; requires the full candidate union. It is an optimistic offline benchmark, not an executable selector.',
           'bootstrap_note': '2000 paired draws over ten overlapping partitions, train/audit seeds and shared held-out record identities; resampling does not make partitions independent.',
           'additional_contrasts': 'Validation or audit versus random, and validation or random versus no-review, were added after inspecting the first completed outputs. Treat as secondary follow-up comparisons.'}
    if dataset == 'wiki':
        for i in deploy:
            stats = []
            for split in range(10):
                pred = matrices[split, :, i]
                gold = truth[split][None, :]
                accuracy = np.mean(pred == gold, axis=1)
                f1 = []
                for c in (0, 1):
                    tp = ((pred == c) & (gold == c)).sum(1)
                    denominator = (pred == c).sum(1) + (gold == c).sum(1)
                    f1.append(np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0))
                stats.append(np.stack((accuracy, f1[0], f1[1]), axis=1))
            values = np.concatenate(stats).mean(0) * 100
            out['policy'][names[i]].update({'accuracy_percent': float(values[0]), 'f1_class0_percent': float(values[1]),
                                            'f1_class1_percent': float(values[2])})
    for left, right in pairs:
        key = left + '_minus_' + right
        delta = (point[:, :, names.index(left)] - point[:, :, names.index(right)]).mean() * 100
        rh = (repaired[:, :, names.index(left)] - repaired[:, :, names.index(right)]).mean()
        out['contrasts'][key] = {'quality_pp': float(delta), 'quality_ci95': np.percentile(boot[key], [2.5, 97.5]).tolist(),
                                  'repairs': float(rh), 'repairs_ci95': np.percentile(boot_repairs[key], [2.5, 97.5]).tolist()}
        if dataset == 'wiki':
            out['contrasts'][key]['accuracy_pp'] = out['policy'][left]['accuracy_percent'] - out['policy'][right]['accuracy_percent']
            out['contrasts'][key]['accuracy_ci95'] = np.percentile(boot_accuracy[key], [2.5, 97.5]).tolist()
    for i in deploy:
        name = names[i]
        if name == 'no_review':
            continue
        oracle_values = point[np.arange(10)[:, None], np.arange(10)[None, :], oracle_indices[:, None]]
        delta = (oracle_values - point[:, :, i]).mean() * 100
        key = 'oracle_minus_' + name
        out['hindsight_oracle'][name] = {'quality_regret_pp': float(delta), 'ci95': np.percentile(boot[key], [2.5, 97.5]).tolist()}
    out['oracle_candidates_by_split'] = [names[i][7:] for i in oracle_indices]
    for field in ('audit_selected', 'validation_selected'):
        out[field + '_counts'] = {m: sum(row[field] == m for row in records) for m in methods}
    return out, records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=('c100', 'wiki'))
    parser.add_argument('learner', choices=('linear', 'mlp'))
    parser.add_argument('--draws', type=int, default=2000)
    args = parser.parse_args()
    result, rows = run(args.dataset, args.learner, args.draws)
    (ROOT / ('equal_cost_' + args.dataset + '_' + args.learner + '_summary.json')).write_text(json.dumps(result, indent=2) + '\n')
    with (ROOT / ('equal_cost_' + args.dataset + '_' + args.learner + '_choices.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2), flush=True)
