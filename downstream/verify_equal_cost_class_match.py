"""Replay matched correction sets and verify test records."""
import argparse, json
from pathlib import Path
import numpy as np

R = Path(__file__).parent


def run(learner, partial=False):
    count = 0
    for split in range(10):
        case = R / 'wiki_attack' / str(split)
        source = json.loads((case / 'choices.json').read_text())['selected']
        with np.load(case / 'queues.npz') as d:
            queue, test = d[source], d['test']
        for rep in range(10):
            seed = 415500 + 10 * rep + split
            path = R / 'equal_cost_class_match' / learner / (str(seed) + '.npz')
            if partial and not path.exists():
                continue
            with np.load(path) as d, np.load(R / 'equal_cost_selection/wiki' / learner / (str(seed) + '.npz')) as b:
                truth, error = d['truth'], b['claim'] != b['reference']
                vq = np.r_[b['validation'], b['pilots'].ravel(), b['validation_review']]
                left0, right0 = np.sort(queue[error[queue]]), np.sort(vq[error[vq]])
                assert np.array_equal(left0, d['source_fixed']) and np.array_equal(right0, d['validation_fixed'])
                assert np.array_equal(d['test_ids'], test) and np.array_equal(d['test_reference'], b['test_reference'])
                common = np.intersect1d(left0, right0)
                left, right = np.setdiff1d(left0, common), np.setdiff1d(right0, common)
                counts = [min(np.sum(truth[left] == c), np.sum(truth[right] == c)) for c in (0, 1)]
                n = int(sum(counts))
                rng = np.random.default_rng(seed + 520000)
                expected = [np.sort(np.r_[common, rng.choice(left, n, replace=False)]),
                            np.sort(np.r_[common, rng.choice(right, n, replace=False)])]
                lset, rset = [common], [common]
                for c, number in enumerate(counts):
                    lset.append(rng.choice(left[truth[left] == c], number, replace=False))
                    rset.append(rng.choice(right[truth[right] == c], number, replace=False))
                expected += [np.sort(np.concatenate(lset)), np.sort(np.concatenate(rset))]
                keys = ('total_source', 'total_validation', 'class_source', 'class_validation')
                assert all(np.array_equal(x, d[k]) for x, k in zip(expected, keys))
                assert len({len(x) for x in expected}) == 1
                assert np.array_equal(np.bincount(truth[expected[2]], minlength=2), np.bincount(truth[expected[3]], minlength=2))
                assert d['predictions'].shape == (4, len(test))
            count += 1
    return count


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--partial', action='store_true')
    a = p.parse_args()
    for learner in ('linear', 'mlp'):
        print(learner, run(learner, a.partial), flush=True)
