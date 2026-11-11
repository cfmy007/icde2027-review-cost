"""Replay paid selections, review budgets and prediction-array completeness."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent


def verify(dataset, learner, partial):
    folder = ROOT / 'equal_cost_selection' / dataset / learner
    count = 0
    for split in range(10):
        if dataset == 'wiki':
            base = ROOT / 'wiki_attack' / str(split)
            source = json.loads((base / 'choices.json').read_text())['selected']
            with np.load(base / 'queues.npz') as d:
                queues = {k: d[k] for k in d.files if k not in ('train', 'test', 'target_ties')}
            expected_sources = {'source': source}
            budget, seedbase = 1200, 415500
        else:
            base = ROOT / 'cifar_candidate_pool/c100/RN18'
            with np.load(base / (str(split) + '.queues.npz')) as d:
                queues = {k: d[k] for k in d.files if k not in ('ids', 'ties', 'truth', 'claims', 'uniform', 'none')}
            with np.load(ROOT / 'official_aum' / str(split) / 'human.npz') as d:
                queues['AUM-raw'] = d['queue']
            with np.load(ROOT / 'cifar_candidate_pool/c100/RN18-with-raw-AUM' / (str(419500 + split) + '.npz')) as d:
                expected_sources = {'source_self': str(d['source_self']), 'source_external': str(d['source_external'])}
            budget, seedbase = 1000, 419500
        for rep in range(10):
            seed = seedbase + 10 * rep + split
            path = folder / (str(seed) + '.npz')
            if partial and not path.exists():
                continue
            with np.load(path) as d:
                names, policies = d['methods'].tolist(), d['policies'].tolist()
                truth, claim = d['reference'], d['claim']
                error = truth != claim
                assert json.loads(str(d['source_methods'])) == expected_sources
                assert int(d['budget']) == budget
                audit, aq = d['audit'], d['audit_review']
                valid, pilots, vq = d['validation'], d['pilots'], d['validation_review']
                assert len(valid) == budget // 10 and pilots.shape == (len(names), (budget // 4) // len(names))
                rng = np.random.default_rng(seed + 510000)
                assert np.array_equal(valid, rng.choice(len(truth), budget // 10, replace=False))
                order = rng.permutation(len(names))
                occupied = np.zeros(len(truth), bool)
                occupied[valid] = True
                positions = np.zeros(len(names), dtype=int)
                replayed = [[] for _ in names]
                for _ in range(pilots.shape[1]):
                    for index in order:
                        queue = queues[names[index]]
                        while occupied[queue[positions[index]]]:
                            positions[index] += 1
                        chosen = int(queue[positions[index]])
                        positions[index] += 1
                        occupied[chosen] = True
                        replayed[index].append(chosen)
                assert np.array_equal(pilots, replayed)
                union = np.unique(np.concatenate([queues[m] for m in names]))
                assert int(d['union_n']) == len(union)
                assert np.array_equal(audit, rng.choice(union, budget // 10, replace=False))
                assert np.array_equal(d['random_review'], rng.choice(len(truth), budget, replace=False))
                reviewed_v = np.r_[valid, pilots.ravel(), vq]
                reviewed_a = np.r_[audit, aq]
                assert len(reviewed_v) == len(np.unique(reviewed_v)) == budget
                assert len(reviewed_a) == len(np.unique(reviewed_a)) == budget
                assert len(np.unique(d['random_review'])) == budget
                scores = np.array([int(error[q[np.isin(q, audit)]].sum()) for q in (queues[m] for m in names)])
                assert np.array_equal(scores, d['audit_scores'])
                assert str(d['audit_selected']) == names[int(np.argmax(scores))]
                assert str(d['validation_selected']) == names[int(np.argmax(d['validation_scores']))]
                marked = np.zeros(len(truth), bool)
                marked[audit] = True
                selected_queue = queues[str(d['audit_selected'])]
                assert np.array_equal(aq, selected_queue[~marked[selected_queue]][:budget - len(audit)])
                marked[:] = False
                marked[np.r_[valid, pilots.ravel()]] = True
                selected_queue = queues[str(d['validation_selected'])]
                assert np.array_equal(vq, selected_queue[~marked[selected_queue]][:budget - marked.sum()])
                checked = dict(expected_sources)
                checked = {key: queues[value] for key, value in checked.items()}
                checked.update({'random': d['random_review'], 'yield_audit': reviewed_a, 'validation_select': reviewed_v,
                                'no_review': np.array([], dtype=int)})
                checked.update({'oracle/' + name: queues[name] for name in names})
                for i, policy in enumerate(policies):
                    assert int(error[checked[policy]].sum()) == int(d['repairs'][i])
                expected_cols = (int(d['n_test']) + 7) // 8 if dataset == 'c100' else int(d['n_test'])
                assert d['predictions'].shape == (len(policies), expected_cols)
                count += 1
    return count


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--partial', action='store_true')
    args = p.parse_args()
    for dataset in ('c100', 'wiki'):
        for learner in ('linear', 'mlp'):
            print(dataset, learner, verify(dataset, learner, args.partial), flush=True)
