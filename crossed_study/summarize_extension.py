"""Build tables and paired summaries for the frozen detector extension."""
import argparse
import json
from pathlib import Path

import pandas as pd

from summarize import avgci
from extend_detectors import CFG, HERE, PROTOCOL

METHODS = {'CNN':'CNN margin','PCA':'PCA margin','RP':'RP margin','CNN-kNN':'CNN kNN','CNN-head':'CNN head'}
MENUS = {'margin':'3 margins','plus_knn':'Add kNN','extended':'Add kNN and head'}


def interval(a, digits=2):
    return f"{a['mean']:.{digits}f} [{a['lo']:.{digits}f}, {a['hi']:.{digits}f}]"


def rows(path, data):
    path.write_text(''.join(' & '.join(line)+r' \\'+ '\n' for line in data),encoding='ascii')


def fold_counts(x):
    return {str(k):int(v) for k,v in x.value_counts().items()}


def summarize(root,paper):
    result = pd.concat([pd.read_csv(root/f'{d}_extended_regret.csv') for d in PROTOCOL['datasets']],ignore_index=True)
    source_review = result[result.strategy=='source_review']
    keys = ['dataset','target','source','source_rate','budget','menu','strategy']
    grouped = {}
    for k,z in result.groupby(keys):
        grouped['/'.join(map(str,k))] = dict(regret=avgci(z.regret_pp),gain_vs_fixed=avgci(z.delta_vs_fixed_cnn_pp),
            utility=avgci(100*z.selected_hits/z.n),selected=fold_counts(z.selected),oracle=fold_counts(z.oracle))
    (root/'extension_summary.json').write_text(json.dumps(grouped,indent=1,allow_nan=False)+'\n')
    g = PROTOCOL['primary']
    primary = source_review[
        (source_review.dataset==g['dataset'])&(source_review.target==g['target'])&
        (source_review.source==g['source'])&(source_review.source_rate==g['rate'])&(source_review.budget==g['budget'])]
    data=[]
    for name,label in MENUS.items():
        z = primary[primary.menu==name]
        detector = '/'.join(f'{METHODS[k]} ({v}/10)' for k,v in fold_counts(z.selected).items())
        data.append([label,detector,interval(avgci(100*z.selected_hits/z.n)),
                     interval(avgci(z.regret_pp)),interval(avgci(z.delta_vs_fixed_cnn_pp))])
    f = result[(result.dataset==g['dataset'])&(result.target==g['target'])&
               (result.budget==g['budget'])&(result.menu=='extended')&(result.strategy=='fixed_cnn')].drop_duplicates('seed')
    data.append(['5 candidates','Fixed CNN margin',interval(avgci(100*f.selected_hits/f.n)),
                 interval(avgci(f.regret_pp)),'0.00'])
    rows(paper/'tables/extension_primary.tex',data)

    z=source_review[(source_review.dataset==g['dataset'])&(source_review.target==g['target'])&(source_review.source==g['source'])]
    sens=z.groupby(['budget','source_rate','menu']).regret_pp.mean()
    table=[]
    for cap in CFG['budgets']:
        cells = ['/'.join(f'{sens.loc[cap,rate,name]:.2f}' for name in MENUS) for rate in CFG['rates']]
        fixed = result[(result.dataset==g['dataset'])&(result.target==g['target'])&(result.budget==cap)&
                       (result.menu=='extended')&(result.strategy=='fixed_cnn')].drop_duplicates('seed')
        table.append([f'{100*cap:.0f}',*cells,f'{fixed.regret_pp.mean():.2f}'])
    rows(paper/'tables/extension_sensitivity.tex',table)

    def cell(dataset,target,source,rate,budget=.1,menu='extended',strategy='source_review'):
        return grouped['/'.join(map(str,(dataset,target,source,rate,budget,menu,strategy)))]

    independent=cell('c100','noisy_label','NC-Aux',.4)
    add_knn=cell('c100','noisy_label','NC-Aux',.4,menu='plus_knn')
    old=cell('c100','noisy_label','NC-Aux',.4,menu='margin')
    independent_lo=cell('c100','noisy_label','NC-Aux',.1)
    same=cell('c100','noisy_label','NC-CNN',.4)
    c10=cell('c10','aggre_label','AQuA-CM-Aux',.1)
    c10_chen=[cell('c10',f'random_label{i}','Chen-ID-hard',.2) for i in (1,2,3)]
    c10_chen_mask=[cell('c10',f'random_label{i}','Chen-ID-mask',.2) for i in (1,2,3)]
    assert all(x['regret']['mean']==0 for x in c10_chen_mask)
    assert independent['selected']=={'CNN-head':10} and independent['regret']['mean']==0
    assert add_knn['selected']=={'CNN-kNN':10} and old['selected']=={'RP':10}
    assert independent_lo['selected']=={'CNN-head':9,'CNN-kNN':1}
    agg_caps = result[(result.dataset=='c10')&(result.budget==.10)&(result.menu=='extended')&
                      (result.strategy=='fixed_cnn')].drop_duplicates(['seed','target'])
    assert len(agg_caps)==50 and (agg_caps.regret_pp==0).all()
    anchors = [
        ['100N noisy','NC-Aux','40',interval(old['regret']),interval(independent['regret'])],
        ['100N noisy','NC-CNN','40',interval(cell('c100','noisy_label','NC-CNN',.4,menu='margin')['regret']),interval(same['regret'])],
        ['10N aggregate','AQuA-CM-Aux','10',interval(cell('c10','aggre_label','AQuA-CM-Aux',.1,menu='margin')['regret']),interval(c10['regret'])],
    ]
    for i,z in enumerate(c10_chen,1):
        anchors.append([f'10N random {i}','Chen hard','20',interval(cell('c10',f'random_label{i}','Chen-ID-hard',.2,menu='margin')['regret']),interval(z['regret'])])
    rows(paper/'tables/extension_public.tex',anchors)
    m = pd.read_csv(root/'c100_extended_metrics.csv')
    h = m.query('side=="target" and protocol=="noisy_label" and budget==.1').groupby('detector').precision.mean()
    cnn_head = h['CNN-head']
    cnn_knn = h['CNN-kNN']
    macros = {
        'ExtensionAbstract': f"On CIFAR-100N, the same independent nearest-class source misses {old['regret']['mean']:.2f} fixes per 100 records among three prototype scores and {add_knn['regret']['mean']:.2f} with a neighbor vote added, then reaches the target optimum when classifier-head confidence is available. Conversely, the ported Chen hard-mask rule incurs {min(x['regret']['mean'] for x in c10_chen):.2f}--{max(x['regret']['mean'] for x in c10_chen):.2f} missed fixes per 100 records on three CIFAR-10N individual-label targets only after the candidate set is expanded.",
        'ExtensionMain': f"Table~\\ref{{tab:extension-primary}} compares the same source and target across candidate sets. Adding kNN makes the source choose kNN over RP on all ten splits, increasing the loss from {interval(old['regret'])} to {interval(add_knn['regret'])} fixes per 100 records. Adding head confidence makes the source choose the head on all ten splits and removes the loss. The head attains target precision {cnn_head:.3f}, while kNN attains {cnn_knn:.3f}. Against the fixed CNN margin policy, choosing from all five improves yield by {interval(independent['gain_vs_fixed'])} fixes per 100 records. The original independent-source penalty is contingent on the available scorers.",
        'ExtensionSensitivity': f"At the full candidate set the independent source has zero regret in 11 of 12 source-rate by budget cells, with {interval(independent_lo['regret'])} at 10\\% corruption and 10\\% review (Table~\\ref{{tab:extension-sensitivity}}). At that cell, nine splits choose the head and one chooses kNN. The same-encoder NC-CNN source retains {interval(same['regret'])} regret at 40\\% corruption and 10\\% review. A fixed CNN margin is target-optimal in all 50 CIFAR-10N split--variant comparisons at that budget, but it misses {f.regret_pp.mean():.2f} fixes per 100 records on CIFAR-100N relative to the head.",
        'ExtensionPublic': f"On CIFAR-10N aggregate, the independent AQuA confusion source incurs {interval(c10['regret'])} fixes per 100 records after expanding the candidate set. On the three individual-label targets, Chen hard selects the local vote in all ten splits and incurs {min(x['regret']['mean'] for x in c10_chen):.2f}--{max(x['regret']['mean'] for x in c10_chen):.2f}, while the shared-mask Chen source has zero regret at the same rates and budget. Table~\\ref{{tab:extension-public}} contrasts these cases with the original candidate set. Public mechanisms can expose a bad choice as the menu grows even when a different source's earlier loss disappears.",
    }
    (paper/'tables/extension_findings.tex').write_text('\n'.join(r'\newcommand{\%s}{%s}'%(k,v) for k,v in macros.items())+'\n',encoding='ascii')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,default=HERE/'results')
    p.add_argument('--paper',type=Path,default=HERE/'paper')
    a=p.parse_args()
    summarize(a.results,a.paper)

