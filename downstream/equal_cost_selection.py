"""Equal-cost source, disagreement-audit, and downstream-validation choices."""
import argparse, json, pickle
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parent
from bridge_encode import encode_pretrained


def macro_f1(y, p, classes):
    values = []
    for k in range(classes):
        tp = np.sum((y == k) & (p == k))
        den = np.sum(y == k) + np.sum(p == k)
        values.append(2 * tp / den if den else 0.)
    return float(np.mean(values))


def fit(x, labels, evaluation, seed, architecture, classes, indices):
    torch.manual_seed(seed)
    if architecture == 'linear':
        model = torch.nn.Linear(x.shape[1], classes).cuda()
    else:
        model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, classes)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.001, fused=True)
    rng = np.random.default_rng(seed)
    for _ in range(18):
        order = torch.as_tensor(rng.permutation(indices), device='cuda')
        for batch in order.split(512):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(x[batch]), labels[batch]).backward()
            optimizer.step()
    with torch.inference_mode():
        return model(evaluation).argmax(1).cpu().numpy().astype(np.uint8)


def prepare(dataset, learner):
    if dataset == 'wiki':
        with np.load(ROOT / 'wiki_attack/base.npz') as d:
            z, claims, reference, testref = (d[k] for k in ('features', 'claims', 'review_reference', 'test_reference'))
        return z, claims, reference, testref, None
    with np.load(ROOT / 'c100_pixels.npz') as d:
        images, y, rest = d['images'], d['y'], d['rest']
    with (ROOT / 'official/test').open('rb') as f:
        raw = pickle.load(f, encoding='bytes')
    test = raw[b'data'].reshape(-1, 3, 32, 32)
    testref = np.asarray(raw[b'fine_labels'], dtype=np.int64)
    train_images = {a.tobytes() for a in images}
    keep = np.array([a.tobytes() not in train_images for a in test])
    test, testref = test[keep], testref[keep]
    assert len(testref) == 9990
    encoder = 'RN18' if learner == 'linear' else 'EB0'
    z = encode_pretrained(encoder, images[rest])
    testz = encode_pretrained(encoder, test)
    claims = np.load(ROOT / 'c100_labels.npz')['noisy_label'][rest]
    return z, claims, y[rest], testref, testz


def run(dataset, learner, splits, output_root):
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    z, all_claims, all_reference, all_testref, testz = prepare(dataset, learner)
    output = output_root / dataset / learner
    output.mkdir(parents=True, exist_ok=True)
    for split in splits:
        if dataset == 'wiki':
            case = ROOT / 'wiki_attack' / str(split)
            source = json.loads((case / 'choices.json').read_text())['selected']
            methods = ['word-confidence', 'char-confidence', 'prototype-margin', 'knn-disagreement', 'native-CL']
            with np.load(case / 'queues.npz') as d:
                ids, test_ids = d['train'], d['test']
                queues = {m: d[m] for m in methods}
            test_features, held = z[test_ids], all_testref[test_ids]
            sources = {'source': source}
            budget, classes, seedbase = 1200, 2, 415500
        else:
            folder = ROOT / 'cifar_candidate_pool/c100/RN18'
            with np.load(folder / (str(split) + '.queues.npz')) as d:
                ids, truth, claim = d['ids'], d['truth'], d['claims']
                queues = {k: d[k] for k in d.files if k not in ('ids', 'ties', 'truth', 'claims', 'uniform', 'none')}
            with np.load(ROOT / 'official_aum' / str(split) / 'human.npz') as d:
                assert np.array_equal(ids, d['ids']) and np.array_equal(claim, d['observed']) and np.array_equal(truth, d['truth'])
                queues['AUM-raw'] = d['queue']
            with np.load(ROOT / 'cifar_candidate_pool/c100/RN18-with-raw-AUM' / (str(419500 + split) + '.npz')) as d:
                methods = d['methods'].tolist()[:-2]
                sources = {'source_self': str(d['source_self']), 'source_external': str(d['source_external'])}
            test_features, held = testz, all_testref
            budget, classes, seedbase = 1000, 100, 419500
        claim, reference = all_claims[ids], all_reference[ids]
        error = claim != reference
        n = len(ids)
        x = torch.as_tensor(z[ids], device='cuda')
        tx = torch.as_tensor(test_features, device='cuda')
        assert all(len(queues[m]) == budget and len(np.unique(queues[m])) == budget for m in methods)
        union = np.unique(np.concatenate([queues[m] for m in methods]))
        for repeat in range(10):
            seed = seedbase + 10 * repeat + split
            dest = output / (str(seed) + '.npz')
            if dest.exists():
                continue
            rng = np.random.default_rng(seed + 510000)
            validation = rng.choice(n, budget // 10, replace=False)
            pilots_per_method = (budget // 4) // len(methods)
            occupied = np.zeros(n, dtype=bool)
            occupied[validation] = True
            pilots = [[] for _ in methods]
            positions = np.zeros(len(methods), dtype=int)
            order = rng.permutation(len(methods))
            for _ in range(pilots_per_method):
                for index in order:
                    q = queues[methods[index]]
                    while occupied[q[positions[index]]]:
                        positions[index] += 1
                    chosen = int(q[positions[index]])
                    positions[index] += 1
                    occupied[chosen] = True
                    pilots[index].append(chosen)
            pilots = np.asarray(pilots, dtype=int)
            assert len(np.unique(np.r_[validation, pilots.ravel()])) == len(validation) + pilots.size
            train_indices = np.flatnonzero(~np.isin(np.arange(n), validation))
            validation_x = x[torch.as_tensor(validation, device='cuda')]
            validation_scores = []
            for i, method in enumerate(methods):
                labels = claim.copy()
                labels[pilots[i]] = reference[pilots[i]]
                pred = fit(x, torch.as_tensor(labels, device='cuda', dtype=torch.long), validation_x, seed, learner, classes, train_indices)
                score = float(np.mean(pred == reference[validation])) if dataset == 'c100' else macro_f1(reference[validation], pred, classes)
                validation_scores.append(score)
            selected_val = methods[int(np.argmax(validation_scores))]
            seen = occupied.copy()
            selected_q = queues[selected_val]
            val_review = selected_q[~seen[selected_q]][:budget - seen.sum()]
            checked_val = np.r_[validation, pilots.ravel(), val_review]
            assert len(checked_val) == budget and len(np.unique(checked_val)) == budget
            audit = rng.choice(union, budget // 10, replace=False)
            marked = np.zeros(n, dtype=bool)
            marked[audit] = True
            audit_scores = [int(error[q[marked[q]]].sum()) for q in (queues[m] for m in methods)]
            selected_audit = methods[int(np.argmax(audit_scores))]
            audit_review = queues[selected_audit][~marked[queues[selected_audit]]][:budget - len(audit)]
            checked_audit = np.r_[audit, audit_review]
            random_review = rng.choice(n, budget, replace=False)
            policies = {name: queues[method] for name, method in sources.items()}
            policies.update({'random': random_review, 'yield_audit': checked_audit, 'validation_select': checked_val, 'no_review': np.array([], dtype=int)})
            if dataset == 'wiki':
                policies.update({'oracle/' + m: queues[m] for m in methods})
            names = list(policies)
            correct, repairs = [], []
            for name in names:
                q = policies[name]
                assert len(q) in (0, budget) and len(np.unique(q)) == len(q)
                lab = claim.copy()
                lab[q] = reference[q]
                repairs.append(int(error[q].sum()))
                pred = fit(x, torch.as_tensor(lab, device='cuda', dtype=torch.long), tx, seed, learner, classes, np.arange(n))
                correct.append(pred == held if dataset == 'c100' else pred)
            if dataset == 'c100':
                oldfolder = 'RN18-with-raw-AUM' if learner == 'linear' else 'RN18-EB0-mlp-with-raw-AUM'
                with np.load(ROOT / 'cifar_candidate_pool/c100' / oldfolder / (str(seed) + '.npz')) as d:
                    assert d['methods'].tolist()[:-2] == methods
                    oracle_preds = np.unpackbits(d['correct'], axis=1, count=len(held))[d['fit_ids'][:len(methods)]].astype(bool)
                    correct.extend(oracle_preds)
                    repairs.extend(d['repairs'][:len(methods)].tolist())
                    names.extend('oracle/' + m for m in methods)
            tmp = dest.with_suffix('.tmp.npz')
            values = np.stack(correct)
            if dataset == 'c100':
                values = np.packbits(values, axis=1)
            np.savez_compressed(tmp, dataset=dataset, learner=learner, split=split, repeat=repeat, seed=seed,
                n_test=len(held), budget=budget, methods=methods, policies=names, repairs=repairs, predictions=values,
                test_reference=held, test_ids=(test_ids if dataset == 'wiki' else np.arange(len(held))),
                source_methods=json.dumps(sources), validation=validation, pilots=pilots, validation_scores=validation_scores,
                validation_selected=selected_val, validation_review=val_review, audit=audit, audit_scores=audit_scores,
                audit_selected=selected_audit, audit_review=audit_review, random_review=random_review, union_n=len(union),
                claim=claim, reference=reference)
            tmp.replace(dest)
            print(dataset, learner, split, repeat, 'val', selected_val, 'audit', selected_audit, 'repairs', repairs[:len(sources)+4], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=('c100', 'wiki'))
    parser.add_argument('learner', choices=('linear', 'mlp'))
    parser.add_argument('--splits', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--out', type=Path, default=ROOT/'retrained')
    args = parser.parse_args()
    run(args.dataset, args.learner, args.splits, args.out)
