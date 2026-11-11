"""Check paper-level result coverage, invariants, and absence of placeholder cells."""
from pathlib import Path
import itertools
import json

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
root=HERE/'results'
cfg=json.loads((HERE/'protocol.json').read_text())
for name,n_prot,n_human in (('hwdb',14,0),('c10',12,5),('c100',12,1)):
    m=pd.read_csv(root/f'{name}_metrics.csv')
    r=pd.read_csv(root/f'{name}_regret.csv')
    assert len(m)==10*4*3*(2*3*n_prot+n_human)
    assert len(r)==10*4*(3*n_prot**2+n_prot*n_human)
    assert set(m.seed)==set(cfg['seeds']) and set(r.seed)==set(cfg['seeds'])
    assert set(m.budget)==set(cfg['budgets']) and set(m.rate)==set(cfg['rates'])
    assert np.max(np.abs(m.auc-m.identity))<1e-10
    assert (m.hits<=m.k).all() and (m.hits<=m.errors).all()
    assert np.allclose(m.gain_pp,100*m.hits/m.n)
    assert np.allclose(m.auto_net_pp,100*(m.auto_fixed-m.auto_harmed)/m.n)
    assert (r.oracle_hits>=r.selected_hits).all()
    assert np.allclose(r.regret_pp,100*(r.oracle_hits-r.selected_hits)/r.n)
    with np.load(root/f'{name}_assignments.npz') as z:
        for seed in cfg['seeds']:
            x=[z[f'{seed}_{key}'] for key in ('ref','source','target')]
            assert len(np.unique(np.concatenate(x)))==sum(map(len,x))
            for side,rate in itertools.product(('source','target'),cfg['rates']):
                t=f'{seed}_{side}_{rate:.2f}_'
                nc=z[t+'NC-CNN']
                mask=z[f'{seed}_{side}_mask'][:round(rate*len(nc))]
                outside=np.ones(len(nc),bool)
                outside[mask]=False
                for g in ('NC-PCA','NC-RP','NC-Aux','AQuA-ID-CNN','AQuA-ID-Aux','AQuA-CM-CNN',
                          'AQuA-CM-Aux','Chen-ID-mask','Uniform','Cyclic'):
                    assert np.array_equal(nc[outside],z[t+g][outside])
    if n_human:
        p=pd.read_csv(root/f'{name}_policies.csv')
        assert (p.human_cost<=p.cap).all()
        assert (p.human_cost==p.audit+p.reviewed).all()
        assert (p.loc[~p.executed,'reviewed']==0).all()
        assert p.loc[~p.executed,'precision'].isna().all()
c=pd.read_csv(root/'legacy_coverage_rows.csv')
assert len(c)==48000
zero=c.loc[~c.executed]
assert len(zero)>0 and zero.precision.isna().all()
assert (zero.reviewed==0).all() and (zero.gain_count==0).all() and (zero.total_cost==zero.audit).all()
d=json.loads((root/'downstream.json').read_text())
assert [r['seed'] for r in d['runs']]==cfg['seeds']
assert len(d['runs'])*len(d['runs'][0]['accuracy'])==50
b=pd.read_csv(root/'transfer_bounds.csv')
assert len(b)==450
assert ((b.auc_gap*b.auc_gap_target)>0)[b.auc_certified].all()
assert ((b.hits_gap*b.hits_gap_target)>0)[b.hits_certified].all()
print('PASS: all study cells, masks, costs, abstentions, regret, AUC identity, and certificates')

from extend_detectors import check_archived_transfer, check_rows, PROTOCOL

for dataset,human_variants in [('c100',1),('c10',5)]:
    new=pd.read_csv(root/f'{dataset}_extended_metrics.csv')
    extension=pd.read_csv(root/f'{dataset}_extended_regret.csv')
    assert set(new.detector)=={'CNN-head','CNN-kNN'}
    assert len(new)==10*4*2*(2*3*12+human_variants)
    assert len(extension)==10*4*3*12*human_variants*3*3
    assert not extension.duplicated(['seed','budget','source_rate','source','target','menu','strategy']).any()
    assert set(extension.seed)==set(cfg['seeds'])
    assert set(extension.source_rate)==set(PROTOCOL['same_source_rates_for_all_human_targets'])
    check_rows(new,extension)
    check_archived_transfer(extension,pd.read_csv(root/f'{dataset}_regret.csv'))
print('PASS: expanded candidate menus, all rates and budgets, fixed CNN, and original transfer identity')
