"""Paired wrong-destination interventions and held-out review decisions."""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import kendalltau
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score

from prepare_features import HERE, CFG, chen_summary, import_script, load_cifar

DETECTORS = ('CNN', 'PCA', 'RP')


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def split_pool(y, idx, rng, n_first):
    first, second = [], []
    for c in np.unique(y):
        group = rng.permutation(idx[y[idx] == c])
        k = int(round(n_first * len(group))) if isinstance(n_first, float) else n_first
        first.extend(group[:k])
        second.extend(group[k:])
    return np.array(first), np.array(second)


def centroids(z, y, ref, classes):
    labels = y[ref]
    return l2(np.stack([z[ref[labels == c]].mean(axis=0) for c in range(classes)]))


def sample_wrong(values, y, u, from_logits=True):
    weights = values.astype(np.float64, copy=True)
    weights[np.arange(len(y)), y] = -np.inf if from_logits else 0.
    if from_logits:
        p = softmax(weights, axis=1)
    else:
        p = weights / weights.sum(axis=1, keepdims=True)
    cumulative = p.cumsum(axis=1)
    # One shared uniform draw per record for every stochastic destination.
    return np.minimum((u[:, None] >= cumulative).sum(axis=1), p.shape[1] - 1)


def wrong_argmax(values, y):
    vals = values.copy()
    vals[np.arange(len(y)), y] = -np.inf
    return vals.argmax(axis=1)


def aqua_confusion(y, pred, eval_y, classes, u):
    counts = np.zeros((classes, classes))
    np.add.at(counts, (y, pred), 1)
    assert np.all(counts.sum(axis=1) > 0)
    cm = np.round(counts / counts.sum(axis=1, keepdims=True), 6)
    # Reproduce the pinned implementation's transpose and EPS smoothing.
    return sample_wrong(cm.T[eval_y] + 1e-7, eval_y, u, from_logits=False)


def partition_order(n, rng):
    # A fixed random row order breaks all score ties without revealing errors.
    return rng.permutation(n)


def prefix_order(scores, ties):
    return ties[np.argsort(-scores[ties], kind='stable')]


def auc_blocks(err, score):
    pos, neg = score[err], score[~err]
    a, b = float(np.mean(pos > 0)), float(np.mean(neg == 0))
    p, q = pos[pos > 0], neg[neg > 0]
    w = None
    if len(p) and len(q):
        w = roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(q))], np.r_[p, q])
    return a, b, w, b * (1 + a) / 2 + a * (1 - b) * (w or 0.)


def metric(y, obs, sim, ties, fractions, clean_margin, detector, protocol):
    err = y != obs
    top = sim.argmax(1)
    score = sim.max(1) - sim[np.arange(len(y)), obs]
    order = prefix_order(score, ties)
    auc = roc_auc_score(err, score)
    a, b, q, reconstructed = auc_blocks(err, score)
    assert abs(auc - reconstructed) < 1e-10
    rows = []
    for frac in fractions:
        k = int(round(frac * len(y)))
        sel = order[:k]
        fix = (err[sel] & (top[sel] == y[sel])).sum()
        harm = (~err[sel] & (top[sel] != y[sel])).sum()
        hits = err[sel].sum()
        rows.append(dict(detector=detector, protocol=protocol, budget=frac, n=len(y), errors=int(err.sum()),
                         auc=auc, ap=average_precision_score(err, score) if not rows else rows[0]['ap'],
                         k=k, hits=int(hits), precision=hits/k, gain_pp=100*hits/len(y),
                         recall=hits/err.sum(), auto_fixed=int(fix), auto_harmed=int(harm),
                         auto_net_pp=100*(fix-harm)/len(y), auto_recall=fix/err.sum(),
                         mean_margin_error=float(score[err].mean()), mean_margin_clean=float(score[~err].mean()),
                         zero_error=1-a, zero_clean=b, q=q, identity= reconstructed,
                         true_nearest_error=float((clean_margin[err] == 0).mean()),
                         true_nearest_clean=float((clean_margin[~err] == 0).mean())))
    return rows, (score, order, err, top)


def budget_policies(y, by_det, rng):
    n = len(y)
    aud_perm = rng.permutation(n)
    for budget, pstar in itertools.product(CFG['budgets'], CFG['precision_targets']):
        cap = int(round(budget * n))
        a = int(np.floor(cap * CFG['audit_share_of_cap']))
        audited = aud_perm[:a]
        audit_mask = np.zeros(n, bool)
        audit_mask[audited] = True
        for detector, (score, order, err, pred) in by_det.items():
            if detector != 'CNN':
                continue
            aorder = order[audit_mask[order]]
            rest = order[~audit_mask[order]]
            hits = np.cumsum(err[aorder])
            # Estimate feasible review fractions on random audit; use a
            # longest initial passing sequence with at least 5 audited records tested.
            chosen = 0
            max_fraction = (cap-a)/(n-a)
            for frac in np.arange(.01, .501, .005):
                if frac > max_fraction + 1e-12:
                    break
                k = int(round(frac*a))
                if k < 5:
                    continue
                if hits[k-1] / k < pstar:
                    break
                chosen = min(cap-a, int(round(frac*(n-a))))
            for policy, sel, aud in (
                ('fixed_review', order[:cap], np.empty(0, dtype=int)),
                ('fixed_auto', order[:cap], np.empty(0, dtype=int)),
                ('calibrated_review', rest[:chosen], audited)):
                got = int(err[sel].sum())
                is_auto = policy == 'fixed_auto'
                damage = int((~err[sel] & (pred[sel] != y[sel])).sum()) if is_auto else 0
                fixed = int((err[sel] & (pred[sel] == y[sel])).sum()) if is_auto else got
                audit_fixed = int(err[aud].sum())
                cost = 0 if is_auto else len(sel) + len(aud)
                precision = got / len(sel) if len(sel) else None
                yield dict(policy=policy, budget=budget, pstar=pstar, n=n, errors=int(err.sum()), cap=cap,
                           audit=len(aud), reviewed=0 if is_auto else len(sel), actions=len(sel),
                           human_cost=cost, human_cost_pct=100*cost/n, executed=bool(len(sel)),
                           precision=precision, met=precision is not None and precision >= pstar,
                           fixed=fixed, audit_fixed=audit_fixed, damage=damage,
                           net_pp=100*(fixed+audit_fixed-damage)/n,
                           recovered=(fixed+audit_fixed)/err.sum(),
                           net_per_human=(fixed+audit_fixed-damage)/cost if cost else None)


def run(dataset, base, cache, out):
    pack = np.load(cache / f'{dataset}_features.npz')
    if dataset == 'hwdb':
        cb = import_script(base / 'scripts/29_curated_bench.py')
        sub = cb.build_named_subset(cb.load_cache(base / 'data/processed/hwdb11_test_32.npz'),
                                    [ch for p in cb.PAIRS for ch in p], 40, 0)
        x = sub['X']
        y = sub['y']
        assert np.array_equal(y, pack['y']) and np.array_equal(sub['writer'], pack['writer'])
        fixed = pack['pair_map']
        font_file = HERE / 'inputs' / 'font_map.npz'
        font = None
        if font_file.exists():
            with np.load(font_file) as data:
                assert np.array_equal(sub['names'], data['names'])
                font = data['pair_map']
        human = {}
        logits, aux_logits = pack['logits'], pack['aux_logits']
    else:
        x, full_y, human = load_cifar(base, dataset)
        assert np.array_equal(full_y, pack['y'])
        rest = pack['rest']
        assert len(np.intersect1d(rest, pack['fit_idx'])) == 0
        x, y = x[rest], full_y[rest]
        human = {key: obs[rest] for key, obs in human.items()}
        logits, aux_logits = pack['cnn_logits'], pack['aux_logits']
    cnn, aux = pack['cnn'], pack['aux']
    temporal = {k: pack[k] for k in ('chen_dest', 'chen_conf')} if 'chen_dest' in pack else chen_summary(pack['avg'], y)
    classes = len(np.unique(y))
    x = x.reshape(len(x), -1).astype(np.float32) / 255.
    rp = l2(x @ np.random.default_rng(0).normal(size=(x.shape[1], 256)).astype(np.float32))
    all_metrics, all_regrets, all_pairs, all_policies, all_gen = [], [], [], [], []
    saved = {}
    for seed in CFG['seeds']:
        rng = np.random.default_rng(seed)
        if dataset == 'hwdb':
            ref, pool = split_pool(y, np.arange(len(y)), rng, 20)
        else:
            ref = np.arange(CFG['cifar_ref_n'])
            pool = np.arange(len(ref), len(y))
        src, test = split_pool(y, pool, rng, .5)
        assert len(np.unique(np.r_[ref, src, test])) == len(y)
        pca = l2(PCA(n_components=50, svd_solver='randomized', random_state=0).fit(x[ref]).transform(x))
        reps = {'CNN': l2(cnn), 'PCA': pca, 'RP': rp, 'Aux': l2(aux)}
        cent = {d: centroids(z, y, ref, classes) for d, z in reps.items()}
        # One random feature-to-class map shared by the two 64-d encoders.
        W = rng.standard_normal((cnn.shape[1], classes)).astype(np.float32)
        saved[f'{seed}_ref'] = ref
        split_metrics = {}
        for side, idx in (('source', src), ('target', test)):
            yt = y[idx]
            sims = {d: z[idx] @ cent[d].T for d, z in reps.items()}
            tied = partition_order(len(idx), rng)
            u = rng.random(len(idx))
            dest = {f'NC-{d}': wrong_argmax(sim, yt) for d, sim in sims.items()}
            dest.update({
                'AQuA-ID-CNN': sample_wrong(cnn[idx] @ W, yt, u),
                'AQuA-ID-Aux': sample_wrong(aux[idx] @ W, yt, u),
                'AQuA-CM-CNN': aqua_confusion(y[ref], logits[ref].argmax(1), yt, classes, u),
                'AQuA-CM-Aux': aqua_confusion(y[ref], aux_logits[ref].argmax(1), yt, classes, u),
                'Chen-ID-mask': temporal['chen_dest'][idx],
                'Uniform': sample_wrong(np.ones((len(idx), classes)), yt, u, False),
                'Cyclic': (yt+1) % classes,
            })
            if dataset == 'hwdb':
                dest['Minimal-pair'] = fixed[yt]
                if font is not None:
                    dest['Font'] = font[yt]
            assert all(np.all(wrong != yt) for wrong in dest.values())
            order_mask = rng.permutation(len(idx))
            saved[f'{seed}_{side}'] = idx
            saved[f'{seed}_{side}_mask'] = order_mask
            saved[f'{seed}_{side}_ties'] = tied
            clean_margin = {d: s.max(1)-s[np.arange(len(yt)), yt] for d, s in sims.items()}
            cases = []
            for rate in CFG['rates']:
                mask = order_mask[:int(round(rate*len(idx)))]
                for g, wrong in dest.items():
                    obs = yt.copy()
                    obs[mask] = wrong[mask]
                    cases.append((g, rate, obs))
                # Original Chen selection rule (transported to the auxiliary
                # CNN and held-out set), included separately from paired masks.
                hard = np.argsort(-temporal['chen_conf'][idx], kind='stable')[:len(mask)]
                obs = yt.copy()
                obs[hard] = dest['Chen-ID-mask'][hard]
                cases.append(('Chen-ID-hard', rate, obs))
            if side == 'target':
                cases.extend((k, v, human[k][idx]) for k, v in CFG['human_matching_rates'].items() if k in human)
            rows = []
            for g, rate, obs in cases:
                changed = obs != yt
                saved[f'{seed}_{side}_{rate:.2f}_{g}'] = obs
                transitions = np.zeros((classes, classes))
                np.add.at(transitions, (yt[changed], obs[changed]), 1)
                modal_share = transitions.max(axis=1).sum() / changed.sum()
                if g in dest and side == 'target':
                    all_gen.append(dict(dataset=dataset, seed=seed, rate=rate, protocol=g,
                                        modal_share=modal_share,
                                        aux_dest_agreement=float((obs[changed] == dest['NC-Aux'][changed]).mean())))
                scores = {}
                for d in DETECTORS:
                    per, scores[d] = metric(yt, obs, sims[d], tied, CFG['budgets'], clean_margin[d], d, g)
                    rows.extend(dict(dataset=dataset, seed=seed, side=side, rate=rate, **m) for m in per)
                if side == 'target' and g in human:
                    arng = np.random.default_rng(seed+650000)
                    all_policies.extend(dict(dataset=dataset, seed=seed, protocol=g, **r)
                                        for r in budget_policies(yt, scores, arng))
            all_metrics.extend(rows)
            split_metrics[side] = pd.DataFrame(rows)
        for rate, budget in itertools.product(CFG['rates'], CFG['budgets']):
            def pivot(side, col):
                m = split_metrics[side]
                m = m[(m.rate == rate) & (m.budget == budget)]
                return m.pivot(index='protocol', columns='detector', values=col)[list(DETECTORS)]
            target_auc = pivot('target', 'auc')
            if budget == CFG['primary_budget']:
                for a, b in itertools.combinations(target_auc.index, 2):
                    x1, x2 = target_auc.loc[a].values, target_auc.loc[b].values
                    inversions = sum((x1[i]-x1[j])*(x2[i]-x2[j]) < 0 for i,j in itertools.combinations(range(3),2))
                    all_pairs.append(dict(dataset=dataset, seed=seed, rate=rate, protocol1=a, protocol2=b,
                                          inversions=inversions, tau=kendalltau(x1,x2).statistic))
            target_hits, source_hits, source_auc = pivot('target', 'hits'), pivot('source','hits'), pivot('source','auc')
            n = len(test)
            for s, t in itertools.product(source_hits.index, target_hits.index):
                selected = source_hits.loc[s].idxmax()
                auc_choice = source_auc.loc[s].idxmax()
                oracle = target_hits.loc[t].idxmax()
                selected_hits = int(target_hits.loc[t, selected])
                best = int(target_hits.loc[t, oracle])
                all_regrets.append(dict(dataset=dataset, seed=seed, rate=rate, budget=budget, source=s, target=t,
                                        n=n, selected=selected, auc_selected=auc_choice, oracle=oracle,
                                        oracle_hits=best, selected_hits=selected_hits, regret_hits=best-selected_hits,
                                        regret_pp=100*(best-selected_hits)/n,
                                        regret_auc_selected_pp=100*(best-target_hits.loc[t,auc_choice])/n,
                                        selected_target_auc=target_auc.loc[t,selected], oracle_target_auc=target_auc.loc[t,oracle]))
        print(f'{dataset} seed {seed} complete', flush=True)
    for label, records in (('metrics', all_metrics), ('regret', all_regrets), ('ranking', all_pairs),
                           ('policies', all_policies), ('mechanism', all_gen)):
        if records:
            pd.DataFrame(records).to_csv(out / f'{dataset}_{label}.csv', index=False)
    np.savez_compressed(out / f'{dataset}_assignments.npz', **saved)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=HERE/'inputs')
    parser.add_argument('--out', type=Path, default=HERE/'results')
    parser.add_argument('dataset', choices=['hwdb', 'c10', 'c100'])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run(args.dataset, args.base, args.cache, args.out)

