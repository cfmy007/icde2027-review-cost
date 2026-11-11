"""Evaluate the finite-pool AUC and top-prefix transfer bounds."""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from cross_protocol import CFG, DETECTORS, centroids, l2, prefix_order
from prepare_features import HERE, import_script, load_cifar

PAIRS = [('AQuA-ID-CNN','AQuA-ID-Aux'), ('AQuA-CM-CNN','AQuA-CM-Aux'),
         ('NC-CNN','NC-Aux'), ('NC-PCA','NC-RP'), ('NC-Aux','Uniform')]


def run(base, cache, results):
    rows = []
    for name in ('hwdb','c10','c100'):
        with np.load(cache/f'{name}_features.npz') as z:
            cnn = l2(z['cnn'])
            if name=='hwdb':
                old = import_script(base/'scripts/29_curated_bench.py')
                sub = old.build_named_subset(old.load_cache(base/'data/processed/hwdb11_test_32.npz'),
                                             [c for pair in old.PAIRS for c in pair],40,0)
                x,y = sub['X'],sub['y']
            else:
                images,labels,_ = load_cifar(base,name)
                x,y = images[z['rest']],labels[z['rest']]
        x = x.reshape(len(x),-1).astype(np.float32)/255
        rp = l2(x @ np.random.default_rng(0).normal(size=(x.shape[1],256)).astype(np.float32))
        c = len(np.unique(y))
        with np.load(results/f'{name}_assignments.npz') as z:
            for seed in CFG['seeds']:
                ref, idx, ties = [z[f'{seed}_{key}'] for key in ('ref','target','target_ties')]
                yt = y[idx]
                pca = l2(PCA(n_components=50,svd_solver='randomized',random_state=0).fit(x[ref]).transform(x))
                sims = {d:r[idx]@centroids(r,y,ref,c).T for d,r in zip(DETECTORS,(cnn,pca,rp))}
                cap = round(.1*len(yt))
                for source,target in PAIRS:
                    ys, yd = [z[f'{seed}_target_0.10_{g}'] for g in (source,target)]
                    err = ys!=yt
                    assert np.array_equal(err,yd!=yt)
                    vals = {}
                    for d,sim in sims.items():
                        ss = sim.max(1)-sim[np.arange(len(yt)),ys]
                        st = sim.max(1)-sim[np.arange(len(yt)),yd]
                        neg = np.sort(ss[~err])
                        def mid(z):
                            return .5*(np.searchsorted(neg,z,'left')+np.searchsorted(neg,z,'right'))/len(neg)
                        bound = np.abs(mid(ss[err])-mid(st[err])).mean()
                        u,v = roc_auc_score(err,ss),roc_auc_score(err,st)
                        a,b = prefix_order(ss,ties)[:cap],prefix_order(st,ties)[:cap]
                        swaps = cap-len(np.intersect1d(a,b))
                        h1,h2 = err[a].sum(),err[b].sum()
                        assert abs(u-v)<=bound+1e-12 and abs(h1-h2)<=swaps
                        vals[d] = (u,v,bound,h1,h2,swaps)
                    for d,e in itertools.combinations(DETECTORS,2):
                        a,b = vals[d],vals[e]
                        rows.append(dict(dataset=name,seed=seed,source=source,target=target,pair=d+'-'+e,
                                         auc_gap=a[0]-b[0],auc_gap_target=a[1]-b[1],auc_bound=a[2]+b[2],
                                         auc_certified=abs(a[0]-b[0])>a[2]+b[2],
                                         hits_gap=a[3]-b[3],hits_gap_target=a[4]-b[4],hits_bound=a[5]+b[5],
                                         hits_certified=abs(a[3]-b[3])>a[5]+b[5]))
    df = pd.DataFrame(rows)
    df.to_csv(results/'transfer_bounds.csv',index=False)
    print(df.groupby(['source','target'])[['auc_certified','hits_certified']].sum().to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=HERE/'inputs')
    p.add_argument('--results',type=Path,default=HERE/'results')
    a=p.parse_args()
    run(a.base,a.cache,a.results)
