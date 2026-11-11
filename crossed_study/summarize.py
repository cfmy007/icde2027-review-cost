"""Generate paper tables, figures and paired split-level summaries from raw rows."""
import argparse
import itertools
from functools import lru_cache
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE/'protocol.json').read_text())
DATASETS = ('hwdb', 'c10', 'c100')
DET = ('CNN', 'PCA', 'RP')
PROTOCOLS = ('NC-CNN', 'NC-PCA', 'NC-RP', 'NC-Aux', 'AQuA-ID-CNN', 'AQuA-ID-Aux',
             'AQuA-CM-CNN', 'AQuA-CM-Aux', 'Chen-ID-mask', 'Chen-ID-hard', 'Uniform', 'Cyclic')
TITLES = {'hwdb':'CASIA-HWDB', 'c10':'CIFAR-10', 'c100':'CIFAR-100'}
HUMANS = {'aggre_label':'10N aggregate', 'random_label1':'10N random 1', 'random_label2':'10N random 2',
          'random_label3':'10N random 3', 'worse_label':'10N worst', 'noisy_label':'100N noisy'}
SHORT = {'AQuA-ID-CNN':'ID-CNN', 'AQuA-ID-Aux':'ID-Aux', 'AQuA-CM-CNN':'CM-CNN', 'AQuA-CM-Aux':'CM-Aux',
         'Chen-ID-mask':'Chen mask', 'Chen-ID-hard':'Chen hard', **HUMANS}
BR = r' \\'
BS = 20000


@lru_cache(maxsize=None)
def bootstrap_indices(n):
    return np.random.default_rng(951).integers(0, n, (BS, n))


def avgci(x):
    a = np.asarray(x, dtype=float)
    if not len(a):
        return dict(n=0, mean=None, lo=None, hi=None)
    b = bootstrap_indices(len(a))
    lo, hi = np.quantile(a[b].mean(axis=1), [.025, .975])
    return dict(n=len(a), mean=float(a.mean()), lo=float(lo), hi=float(hi))


def triple(row, keys, decimals):
    return '/'.join(f'{row[x]:.{decimals}f}' for x in keys)


def write(path, lines):
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def heatmap(ax, mat, title, vmax, show_y=True):
    im = ax.imshow(mat, cmap='YlOrBr', vmin=0, vmax=vmax, aspect='auto')
    ax.set_xticks(range(mat.shape[1]), [SHORT.get(k,k) for k in mat.columns], rotation=60, ha='right')
    ax.set_yticks(range(len(mat.index)), [SHORT.get(k,k) for k in mat.index] if show_y else ['']*len(mat.index))
    ax.set_title(title, fontsize=10)
    for i,j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
        v = mat.iloc[i,j]
        label = f'{v:.0f}' if v >= .5 else (r'$<$1' if v > 0 else '.')
        ax.text(j,i,label,ha='center',va='center',fontsize=7,color='white' if v>vmax*.65 else 'black')
    return im


def load(root, label):
    return pd.concat([pd.read_csv(root/f'{d}_{label}.csv') for d in DATASETS
                      if (root/f'{d}_{label}.csv').exists()], ignore_index=True)


def main(root, paper, legacy):
    tab, fig = paper/'tables', paper/'figures'
    tab.mkdir(parents=True,exist_ok=True)
    fig.mkdir(parents=True,exist_ok=True)
    metrics, regret, ranks, policies, mechanism = [load(root,x) for x in ('metrics','regret','ranking','policies','mechanism')]
    target = metrics.query('side=="target" and budget==.1')
    synth = target.query('rate==.1')
    ag = synth.groupby(['dataset','protocol','detector'])[['auc','ap','precision']].mean()
    latex = []
    for g in PROTOCOLS + ('Minimal-pair','Font'):
        cells = []
        for d in DATASETS:
            if (d,g) not in ag.index:
                cells.extend(['--','--','--'])
                continue
            t = ag.loc[d,g]
            cells.extend(triple(t[k],DET,3 if k=='auc' else 2) for k in ('auc','ap','precision'))
        latex.append(' & '.join([g]+cells)+BR)
    write(tab/'matrix.tex',latex)
    human = target[target.protocol.isin(HUMANS)]
    hm = human.groupby(['protocol','detector'])[['auc','ap','precision','gain_pp','auto_net_pp','errors','n']].mean()
    rr = regret.query('budget==.1')
    hreg = rr[rr.target.isin(HUMANS)]
    selected = hreg[hreg.source.isin(['NC-CNN','NC-Aux','Chen-ID-hard'])]
    latex = []
    for k, label in HUMANS.items():
        z = hm.loc[k]
        by_s = selected[selected.target==k].groupby('source').regret_pp.mean()
        latex.append(' & '.join([label,f"{100*z.errors.iloc[0]/z.n.iloc[0]:.1f}",
            triple(z.auc,DET,3),triple(z.ap,DET,3),triple(z.precision,DET,3),
            triple(z.auto_net_pp,DET,2), f'{by_s["NC-CNN"]:.2f}', f'{by_s["NC-Aux"]:.2f}', f'{by_s["Chen-ID-hard"]:.2f}'])+BR)
    write(tab/'human.tex',latex)
    lines = []
    for d, rate in itertools.product(DATASETS,CFG['rates']):
        z = target[(target.dataset==d)&(target.rate==rate)]
        vals = []
        for g in ('NC-PCA','NC-RP'):
            a = z[z.protocol==g].pivot(index='seed',columns='detector',values='auc')
            ci = avgci(a.PCA-a.RP)
            vals.append(f"{ci['mean']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}]")
        lines.append(' & '.join([TITLES[d], f'{rate:.0%}'.replace('%',r'\%'), *vals])+BR)
    write(tab/'rank_gaps.tex',lines)
    cov = pd.read_csv(root/'legacy_coverage_summary.csv').query('target==.7 and margin==0')
    latex = []
    for d, z in cov.groupby('dataset', sort=True):
        for r in z.sort_values('alpha', ascending=False).itertuples():
            latex.append(' & '.join([d.split(' ')[1] if '10N' in d else '100N',
                        f'{r.alpha:.2f}',f'{r.execution_rate:.3f}',f'{r.precision_cond:.3f}',
                        f'{r.met_conditional:.3f}',f'{r.met_unconditional:.3f}',
                        f'{r.gain_pp_probe:.2f}',f'{r.total_cost_pct:.1f}'])+BR)
    write(tab/'coverage.tex',latex)
    pol = policies.query('pstar==.7 and budget==.1')
    latex = []
    for k in HUMANS:
        z = pol[pol.protocol==k]
        g = z.groupby('policy')
        fixed = g.get_group('fixed_review')
        calibrated = g.get_group('calibrated_review')
        auto = g.get_group('fixed_auto')
        cp = calibrated.loc[calibrated.executed, 'precision'].mean()
        latex.append(' & '.join([HUMANS[k],f'{fixed.net_pp.mean():.2f}', f'{fixed.precision.mean():.3f}', f'{fixed.met.mean():.2f}',
            f'{calibrated.net_pp.mean():.2f}',f'{calibrated.human_cost_pct.mean():.2f}',
            f'{calibrated.executed.mean():.2f}', f'{cp:.3f}' if np.isfinite(cp) else '--',
            f'{calibrated.met.mean():.2f}', f'{auto.net_pp.mean():.2f}'])+BR)
    write(tab/'policies.tex',latex)
    protocols = ['NC-CNN','NC-PCA','NC-RP','NC-Aux','AQuA-ID-CNN','AQuA-ID-Aux',
                 'AQuA-CM-CNN','AQuA-CM-Aux','Chen-ID-mask','Chen-ID-hard','Uniform']
    mats = []
    for d in DATASETS:
        z = rr.query('dataset==@d and rate==.1')
        mats.append(100*z.pivot_table(index='source',columns='target',values='regret_pp').reindex(index=protocols,columns=protocols))
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
    f, ax = plt.subplots(1,3,figsize=(10,3.95),layout='constrained')
    vmax = max(m.max().max() for m in mats)
    for i,(d,m) in enumerate(zip(DATASETS,mats)):
        im = heatmap(ax[i],m,TITLES[d],vmax,show_y=i==0)
        ax[i].set_xlabel('Target injection protocol')
    ax[0].set_ylabel('Selection protocol')
    f.colorbar(im,ax=ax,label='Missed fixes / 10,000 target records',shrink=.75)
    f.savefig(fig/'regret.pdf',bbox_inches='tight')
    plt.close(f)
    f, axes = plt.subplots(1,3,figsize=(9.6,2.65),layout='constrained')
    for ax,k in zip(axes,('aggre_label','worse_label','noisy_label')):
        zz = policies[(policies.pstar==.7)&(policies.protocol==k)]
        for i,(policy,name) in enumerate((('fixed_review','Fixed review'),('calibrated_review','Calibrated review'),
                                          ('fixed_auto','Auto (0 added checks)'))):
            z = zz[zz.policy==policy].groupby('budget').net_pp.mean()
            ax.plot(100*z.index,z,marker=['o','s','^'][i],label=name)
        ax.axhline(0,lw=.6,color='.65')
        ax.set_title(HUMANS[k])
        ax.set_xlabel('Action cap (% of records)')
    axes[0].set_ylabel('Net fixes / 100 records')
    axes[-1].legend(fontsize=6,loc='upper left')
    f.savefig(fig/'budgets.pdf',bbox_inches='tight')
    plt.close(f)
    legacy_rule = json.loads((legacy/'protocol_rules.json').read_text(encoding='utf-8'))
    old = pd.DataFrame(legacy_rule['rows'])
    arch = old.pivot_table(index='injection',columns='repr',values='proto_margin')
    for g in ('fixed','neighbor','font','source_pca','circular'):
        line = arch.loc[g]
        tag = g.replace("_","-")
        latex = f"{tag} & {line['CNN-64']:.3f} & {line['PCA-50']:.3f} & {line['RP-256']:.3f} & {line['CNN-64']-line['PCA-50']:.3f}"
        with (tab/'legacy.tex').open('w' if g=='fixed' else 'a') as stream:
            stream.write(latex+BR+'\n')
    out = {}
    for keys,z in target.groupby(['dataset','protocol','rate','detector']):
        out['metric:'+'/'.join(map(str,keys))] = {k:avgci(z[k]) for k in
            ['auc','ap','precision','gain_pp','auto_net_pp','zero_error','true_nearest_error',
             'true_nearest_clean','mean_margin_error','mean_margin_clean']}
    for keys,z in rr.groupby(['dataset','source','target','rate']):
        out['regret:'+'/'.join(map(str,keys))] = dict(review=avgci(z.regret_pp),
            auc_choice=avgci(z.regret_auc_selected_pp), selected=z.selected.value_counts().to_dict(),
            oracle=z.oracle.value_counts().to_dict(), hits=avgci(z.regret_hits),
            selected_auc=avgci(z.selected_target_auc), oracle_auc=avgci(z.oracle_target_auc))
    for keys,z in ranks.groupby(['dataset','protocol1','protocol2','rate']):
        out['rank:'+'/'.join(map(str,keys))] = dict(tau=avgci(z.tau.dropna()),
            inversions=avgci(z.inversions), any_inversion=int((z.inversions>0).sum()),
            inversions_sum=int(z.inversions.sum()))
    for (d,rate,g), z in target.groupby(['dataset','rate','protocol']):
        auc = z.pivot(index='seed',columns='detector',values='auc')
        for i,j in itertools.combinations(DET,2):
            out[f'gap:{d}/{rate}/{g}/{i}-{j}'] = avgci(auc[i]-auc[j])
    for (d,rate),z in target.groupby(['dataset','rate']):
        auc = z.pivot(index='seed',columns=['protocol','detector'],values='auc')
        for i,j in itertools.combinations(DET,2):
            self_pair = (auc[f'NC-{i}',i]-auc[f'NC-{j}',i])-(auc[f'NC-{i}',j]-auc[f'NC-{j}',j])
            out[f'interaction:{d}/{rate}/{i}-{j}'] = avgci(self_pair)
        for family in ('NC','AQuA-ID','AQuA-CM'):
            for detector in DET:
                diff = auc[family+'-CNN',detector]-auc[family+'-Aux',detector]
                out[f'source_swap:{d}/{rate}/{family}/{detector}'] = avgci(diff)
        for detector in DET:
            out[f'hard_mask:{d}/{rate}/{detector}'] = avgci(auc['Chen-ID-hard',detector]-auc['Chen-ID-mask',detector])
    for keys,z in pol.groupby(['dataset','protocol']):
        pols = z.pivot(index='seed',columns='policy',values='net_pp')
        out['policy:'+'/'.join(keys)] = avgci(pols.calibrated_review-pols.fixed_review)
    for keys,z in mechanism.groupby(['dataset','rate','protocol']):
        out['mechanism:'+'/'.join(map(str,keys))] = {k:avgci(z[k]) for k in ['modal_share','aux_dest_agreement']}
    for f,key in ((legacy/'endtoend_more.json','archived'),(root/'downstream.json','new')):
        if not f.exists():
            continue
        raw = json.loads(f.read_text())
        if key=='archived':
            for setting,runs in raw.items():
                z = pd.DataFrame(r['acc'] for r in runs)
                for method in ('fixed10','calib','review'):
                    out[f'downstream:{key}/{setting}/{method}'] = avgci(100*(z[method]-z.none))
        else:
            z = pd.DataFrame(r['accuracy'] for r in raw['runs'])
            for method in ('auto','fix_only','damage_only','review'):
                out[f'downstream:{key}/{method}'] = avgci(100*(z[method]-z.none))
            out['downstream:new/auto-minus-fix'] = avgci(100*(z.auto-z.fix_only))
            out['downstream:new/interaction'] = avgci(100*(z.auto-z.fix_only-z.damage_only+z.none))
            out['downstream:new/design'] = {k:v for k,v in raw.items() if k!='runs'}
            lines = []
            for method in ('none','auto','fix_only','damage_only','review'):
                a = avgci(100*(z[method]-z.none))
                lines.append(' & '.join([method.replace('_',' '),str(raw['net_labels'][method]),
                    f'{100*z[method].mean():.2f}',f"{a['mean']:+.2f} [{a['lo']:+.2f}, {a['hi']:+.2f}]"])+BR)
            write(tab/'downstream.tex',lines)
    (root/'paired_summary.json').write_text(json.dumps(out,indent=1,allow_nan=False),encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results',type=Path,default=HERE/'results')
    p.add_argument('--paper',type=Path,required=True)
    p.add_argument('--legacy',type=Path,default=HERE/'legacy')
    a = p.parse_args()
    main(a.results,a.paper,a.legacy)


