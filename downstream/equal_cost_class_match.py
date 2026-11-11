"""Matched total, shared and verified-class correction interventions on Wiki."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from equal_cost_selection import fit

R = Path(__file__).parent


def run(learner):
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    with np.load(R / 'wiki_attack/base.npz') as d:
        features, claims, reference, testref = (d[k] for k in ('features', 'claims', 'review_reference', 'test_reference'))
    out = R / 'equal_cost_class_match' / learner
    out.mkdir(parents=True, exist_ok=True)
    for split in range(10):
        case = R / 'wiki_attack' / str(split)
        source = json.loads((case / 'choices.json').read_text())['selected']
        with np.load(case / 'queues.npz') as d:
            train, test, source_queue = d['train'], d['test'], d[source]
        x = torch.as_tensor(features[train], device='cuda')
        tx = torch.as_tensor(features[test], device='cuda')
        observed, truth = claims[train], reference[train]
        error = observed != truth
        for rep in range(10):
            seed = 415500 + 10 * rep + split
            path = out / (str(seed) + '.npz')
            if path.exists():
                continue
            with np.load(R / 'equal_cost_selection/wiki' / learner / (str(seed) + '.npz')) as d:
                validation_queue = np.r_[d['validation'], d['pilots'].ravel(), d['validation_review']]
            source_fixed = np.sort(source_queue[error[source_queue]])
            validation_fixed = np.sort(validation_queue[error[validation_queue]])
            common = np.intersect1d(source_fixed, validation_fixed)
            left = np.setdiff1d(source_fixed, common)
            right = np.setdiff1d(validation_fixed, common)
            counts = [min(np.sum(truth[left] == c), np.sum(truth[right] == c)) for c in (0, 1)]
            n = int(sum(counts))
            rng = np.random.default_rng(seed + 520000)
            total_left = np.sort(np.r_[common, rng.choice(left, n, replace=False)])
            total_right = np.sort(np.r_[common, rng.choice(right, n, replace=False)])
            matched_left, matched_right = [common], [common]
            for c, count in enumerate(counts):
                matched_left.append(rng.choice(left[truth[left] == c], count, replace=False))
                matched_right.append(rng.choice(right[truth[right] == c], count, replace=False))
            class_left = np.sort(np.concatenate(matched_left))
            class_right = np.sort(np.concatenate(matched_right))
            sets = [total_left, total_right, class_left, class_right]
            assert len({len(a) for a in sets}) == 1
            assert np.array_equal(np.bincount(truth[class_left], minlength=2), np.bincount(truth[class_right], minlength=2))
            assert np.array_equal(np.intersect1d(total_left, total_right), common)
            assert np.array_equal(np.intersect1d(class_left, class_right), common)
            predictions = []
            for fixed in sets:
                labels = observed.copy()
                labels[fixed] = truth[fixed]
                pred = fit(x, torch.as_tensor(labels, device='cuda', dtype=torch.long), tx, seed, learner, 2, np.arange(len(train)))
                predictions.append(pred)
            tmp = path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, seed=seed, split=split, repeat=rep, test_ids=test, test_reference=testref[test],
                                truth=truth, source_fixed=source_fixed, validation_fixed=validation_fixed,
                                total_source=total_left, total_validation=total_right, class_source=class_left,
                                class_validation=class_right, predictions=np.stack(predictions))
            tmp.replace(path)
            print(learner, split, rep, 'total', len(total_left), 'common', len(common), 'class_counts', counts, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('learner', choices=('linear', 'mlp'))
    run(p.parse_args().learner)
