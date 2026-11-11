"""Describe verified correction composition for each paid policy."""
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent


def run(dataset, learner):
    rows = []
    base_seed = 419500 if dataset == 'c100' else 415500
    for split in range(10):
        if dataset == 'c100':
            with np.load(ROOT / 'cifar_candidate_pool/c100/RN18' / (str(split) + '.queues.npz')) as d:
                queues = {k: d[k] for k in d.files if k not in ('ids', 'ties', 'truth', 'claims', 'uniform', 'none')}
            with np.load(ROOT / 'official_aum' / str(split) / 'human.npz') as d:
                queues['AUM-raw'] = d['queue']
            classes = 100
        else:
            with np.load(ROOT / 'wiki_attack' / str(split) / 'queues.npz') as d:
                queues = {k: d[k] for k in d.files if k not in ('train', 'test', 'target_ties')}
            classes = 2
        for rep in range(10):
            seed = base_seed + 10 * rep + split
            with np.load(ROOT / 'equal_cost_selection' / dataset / learner / (str(seed) + '.npz')) as d:
                claim, truth = d['claim'], d['reference']
                choices = json.loads(str(d['source_methods']))
                policies = {name: queues[m] for name, m in choices.items()}
                policies['random'] = d['random_review']
                policies['yield_audit'] = np.r_[d['audit'], d['audit_review']]
                policies['validation_select'] = np.r_[d['validation'], d['pilots'].ravel(), d['validation_review']]
                for name, q in policies.items():
                    changed = q[claim[q] != truth[q]]
                    updated = np.bincount(claim, minlength=classes) - np.bincount(claim[changed], minlength=classes) + np.bincount(truth[changed], minlength=classes)
                    fixed_by_class = np.bincount(truth[changed], minlength=classes)
                    row = {'dataset': dataset, 'learner': learner, 'split': split, 'repeat': rep, 'policy': name,
                           'corrected': len(changed), 'represented_corrected_classes': int((fixed_by_class > 0).sum())}
                    if dataset == 'wiki':
                        row.update({'corrected_1_to_0': int(np.sum((claim[changed] == 1) & (truth[changed] == 0))),
                                    'corrected_0_to_1': int(np.sum((claim[changed] == 0) & (truth[changed] == 1))),
                                    'train_positive_fraction': float(updated[1] / len(claim))})
                    else:
                        row['max_corrected_class_fraction'] = float(fixed_by_class.max() / max(len(changed), 1))
                        row['label_total_variation_from_reference'] = float(np.abs(updated / len(claim) - np.bincount(truth, minlength=classes) / len(claim)).sum() / 2)
                    rows.append(row)
    path = ROOT / ('equal_cost_' + dataset + '_' + learner + '_composition.csv')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    out = {}
    for policy in sorted(set(row['policy'] for row in rows)):
        subset = [row for row in rows if row['policy'] == policy]
        out[policy] = {key: float(np.mean([r[key] for r in subset])) for key in subset[0] if key not in ('dataset', 'learner', 'split', 'repeat', 'policy')}
    (ROOT / ('equal_cost_' + dataset + '_' + learner + '_composition.json')).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=('c100', 'wiki'))
    parser.add_argument('learner', choices=('linear', 'mlp'))
    a = parser.parse_args()
    run(a.dataset, a.learner)
