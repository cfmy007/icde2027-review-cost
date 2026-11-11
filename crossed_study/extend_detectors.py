"""Score non-prototype baselines on frozen claims and replay human transfer."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from cross_protocol import CFG, l2, prefix_order
from prepare_features import HERE, load_cifar

PROTOCOL = json.loads((HERE/'extension_protocol.json').read_text())
CANDIDATES = PROTOCOL['candidate_sets']


def label_disagreement(features, labels, reference, k):
    index = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
    index.fit(features[reference])
    neighbors = index.kneighbors(features, return_distance=False)
    return labels[reference[neighbors]]


def new_metrics(dataset, base, cache, original):
    _, clean, human = load_cifar(base,dataset)
    with np.load(cache/f'{dataset}_features.npz') as z:
        assert np.array_equal(z['y'],clean)
        idx, embeddings = z['rest'], l2(z['cnn'])
        logits = PROTOCOL['head_logit_scale']*z['cnn_logits'].astype(np.float64)
    y = clean[idx]
    human = {k:v[idx] for k,v in human.items()}
    lognorm = logsumexp(logits,axis=1)
    out = []
    with np.load(original/f'{dataset}_assignments.npz') as a:
        ref = a[f"{CFG['seeds'][0]}_ref"]
        neighbors = label_disagreement(embeddings,y,ref,PROTOCOL['knn_neighbors'])
        for seed in CFG['seeds']:
            assert np.array_equal(ref,a[f'{seed}_ref'])
            for side in ('source','target'):
                records = a[f'{seed}_{side}']
                assert not np.intersect1d(ref,records).size
                true = y[records]
                ties = a[f'{seed}_{side}_ties']
                scope = f'{seed}_{side}_'
                claims = [k for k in a.files if k.startswith(scope) and k[len(scope):len(scope)+1].isdigit()]
                for key in claims:
                    token, name = key[len(scope):].split('_',1)
                    rate = float(token)
                    obs = a[key]
                    if name in human:
                        assert side=='target' and np.array_equal(obs,human[name][records])
                    err = true!=obs
                    scores = {'CNN-kNN': (neighbors[records]!=obs[:,None]).sum(axis=1),
                              'CNN-head': lognorm[records]-logits[records,obs]}
                    for method, score in scores.items():
                        order = prefix_order(score,ties)
                        auc, ap = roc_auc_score(err,score),average_precision_score(err,score)
                        hits = np.cumsum(err[order])
                        for budget in CFG['budgets']:
                            cap = round(budget*len(obs))
                            h = int(hits[cap-1])
                            out.append(dict(dataset=dataset,seed=seed,side=side,rate=rate,protocol=name,
                                            detector=method,budget=budget,n=len(obs),k=cap,errors=int(err.sum()),
                                            auc=auc,ap=ap,hits=h,precision=h/cap,gain_pp=100*h/len(obs)))
    return pd.DataFrame(out)


def transfer(data):
    keys = ['seed','budget','rate','protocol']
    src = data[data.side=='source'].pivot(index=keys,columns='detector',values='hits')
    source_auc = data[data.side=='source'].pivot(index=keys,columns='detector',values='auc')
    t = data[(data.side=='target') & data.protocol.isin(CFG['human_matching_rates'])]
    target = t.pivot(index=['seed','budget','protocol'],columns='detector',values='hits')
    target_n = t.groupby(['seed','budget','protocol']).n.first()
    n_choices = len(CANDIDATES['extended'])
    assert src.shape[1]==target.shape[1]==n_choices
    target_by_seed = {k:list(z.index.get_level_values('protocol')) for k,z in target.groupby(level=['seed','budget'])}
    rows = []
    for (seed,budget,rate,source), sr in src.iterrows():
        for name, methods in CANDIDATES.items():
            choices = {
                'source_review':sr[methods].idxmax(),
                'source_auc':source_auc.loc[(seed,budget,rate,source),methods].idxmax(),
                'fixed_cnn':PROTOCOL['fixed_baseline'],
            }
            for human in target_by_seed[(seed,budget)]:
                th = target.loc[(seed,budget,human),methods]
                oracle = th.idxmax()
                n = target_n.loc[(seed,budget,human)]
                for strategy, detector in choices.items():
                    selected = int(th[detector])
                    best = int(th[oracle])
                    fixed_cnn = int(th['CNN'])
                    rows.append(dict(dataset=data.dataset.iloc[0],seed=seed,budget=budget,source_rate=rate,
                                     source=source,target=human,menu=name,strategy=strategy,selected=detector,
                                     oracle=oracle,n=n,selected_hits=selected,oracle_hits=best,fixed_cnn_hits=fixed_cnn,
                                     regret_hits=best-selected,regret_pp=100*(best-selected)/n,
                                     delta_vs_fixed_cnn_pp=100*(selected-fixed_cnn)/n))
    return pd.DataFrame(rows)


def check_archived_transfer(new, old):
    key = ['seed','budget','source_rate','source','target']
    z = new[(new.menu=='margin') & (new.strategy=='source_review')]
    baseline = old[old.target.isin(CFG['human_matching_rates'])].rename(columns={'rate':'source_rate'})
    assert len(baseline)>0
    cols = ['regret_hits','selected_hits','oracle_hits','selected','oracle']
    paired = z.merge(baseline,on=key,suffixes=('_new','_old'),validate='one_to_one')
    assert len(paired)==len(baseline)
    for c in cols:
        assert (paired[c+'_new']==paired[c+'_old']).all(), c


def check_rows(metrics, regret):
    assert (metrics.hits<=metrics.k).all() and (metrics.hits<=metrics.errors).all()
    assert (regret.regret_hits>=0).all()
    assert (regret[regret.strategy=='fixed_cnn'].delta_vs_fixed_cnn_pp==0).all()
    assert np.allclose(regret.regret_pp,100*regret.regret_hits/regret.n)
    assert np.allclose(regret.delta_vs_fixed_cnn_pp,100*(regret.selected_hits-regret.fixed_cnn_hits)/regret.n)


def run(base,cache,original,out):
    out.mkdir(exist_ok=True,parents=True)
    for dataset in PROTOCOL['datasets']:
        extra = new_metrics(dataset,base,cache,original)
        prior = pd.read_csv(original/f'{dataset}_metrics.csv')[extra.columns]
        raw = pd.concat([prior,extra],ignore_index=True)
        regret = transfer(raw)
        check_archived_transfer(regret,pd.read_csv(original/f'{dataset}_regret.csv'))
        check_rows(extra,regret)
        extra.to_csv(out/f'{dataset}_extended_metrics.csv',index=False)
        regret.to_csv(out/f'{dataset}_extended_regret.csv',index=False)
        print(f'{dataset}: {len(extra)} new score/budget rows; {len(regret)} transfer rows; archived choices matched',flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=HERE/'inputs')
    p.add_argument('--original',type=Path,default=HERE/'results')
    p.add_argument('--out',type=Path,default=HERE/'results')
    a=p.parse_args()
    run(a.base,a.cache,a.original,a.out)
