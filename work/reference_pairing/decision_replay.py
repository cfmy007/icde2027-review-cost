from pathlib import Path
import csv, json
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'strong_eab_evidence'
OUT = Path(__file__).resolve().parent

def interleave(a, b):
    result, seen = [], set()
    for x,y in zip(a,b):
        for v in (int(x),int(y)):
            if v not in seen:
                seen.add(v); result.append(v)
                if len(result)==len(a): return np.array(result)
    raise RuntimeError('incomplete queue')

def audit(a, b, errors, rng, gated):
    k = len(a)
    left, right = np.setdiff1d(a,b),np.setdiff1d(b,a)
    if gated and len(left) <= int(.10*k):
        return int(errors[a].sum()),0,False
    sample = min(int(.05*k),len(left))
    if sample == 0: return int(errors[a].sum()),0,False
    x,y=rng.choice(left,sample,False),rng.choice(right,sample,False)
    chosen=b if errors[y].mean()>errors[x].mean() else a
    seen=np.concatenate((x,y))
    review=chosen[~np.isin(chosen,seen)][:k-len(seen)]
    return int(errors[seen].sum()+errors[review].sum()),len(seen),bool(chosen is b)

def main():
    rows=[]
    for dataset, folder in [('fashion-RN18','fashion/RN18'),('fashion-EB0','fashion/EB0'),('allenoise','allenoise'),('cifar100-RN18',''),('cifar100-EB0','')]:
        for split in range(10):
            if dataset.startswith('cifar100'):
                arch=dataset.split('-')[1]
                with (ROOT.parent/'strong_eab'/'paid_c100.csv').open(newline='') as f:
                    c=next(r for r in csv.DictReader(f) if r['architecture']==arch and int(r['split'])==split)
                am,bm=c['self_method'],c['external_method']
                paths=[ROOT.parent/'paperB_integrated/artifact/followup/results/result_c100'/f'c100-{arch}-{split}-native-queues.npz',ROOT/'public_methods'/arch/str(split)/'queues.npz']
                with np.load(ROOT/'official_aum'/str(split)/'human.npz') as z:
                    errors=z['truth']!=z['observed']
                with np.load(paths[0]) as native,np.load(paths[1]) as public:
                    a=(native if am.startswith('native-') else public)['human-probe/'+am][:1000]
                    b=(native if bm.startswith('native-') else public)['human-probe/'+bm][:1000]
                assert int(errors[a].sum())==int(c['self_hits']) and int(errors[b].sum())==int(c['external_hits'])
            else:
                case=ROOT/folder/str(split)
                with (case/'choices.csv').open(newline='') as f:
                    choices={(r['source'],r['source_purity']):r['selected'] for r in csv.DictReader(f) if r['target_purity']=='human'}
                am,bm=choices['self-10','clean'],choices['external-10','human']
                name='data.npz' if dataset=='allenoise' else 'queues.npz'
                with np.load(case/name) as z:
                    target=z['target']; errors=z['truth'][target]!=z['claims'][target]
                    a=z['human/'+am];b=z['human/'+bm]
            assert len(b)==len(a) and len(np.unique(a))==len(a) and len(np.unique(b))==len(b)
            for direction in ('self_default','external_default'):
                first,second,ma,mb=(a,b,am,bm) if direction=='self_default' else (b,a,bm,am)
                for fraction in (0.25,0.5,1.0):
                    k=round(len(first)*fraction);aa,bb=first[:k],second[:k]
                    ha,hb=int(errors[aa].sum()),int(errors[bb].sum()); mixed=int(errors[interleave(aa,bb)].sum())
                    exclusive=len(np.setdiff1d(aa,bb)); assert abs(ha-hb)<=exclusive
                    for rep in range(200):
                        seed=764000+1000*split+rep
                        ah, cost, switch=audit(aa,bb,errors,np.random.default_rng(seed),False)
                        gh, gcost, gswitch=audit(aa,bb,errors,np.random.default_rng(seed),True)
                        rows.append(dict(dataset=dataset,direction=direction,split=split,rep=rep,budget=k,budget_fraction=fraction,exclusive=exclusive,
                            default_method=ma,alt_method=mb,
                            default_hits=ha,alternate_hits=hb,interleave_hits=mixed,audit_hits=ah,audit_cost=cost,audit_switch=switch,
                            gated_hits=gh,gated_cost=gcost,gated_switch=gswitch))
    groups=defaultdict(list)
    for r in rows: groups[r['dataset'],r['direction'],r['budget_fraction']].append(r)
    summary=[]; rng=np.random.default_rng(764999)
    for (dataset,direction,fraction),rs in groups.items():
        d={'dataset':dataset,'direction':direction,'budget':rs[0]['budget'],'budget_fraction':fraction,'mean_exclusive':np.mean([r['exclusive'] for r in rs]),
           'small_disagreement_splits':len({r['split'] for r in rs if r['exclusive']<=.1*r['budget']}),
           'mean_audit_spend':np.mean([r['audit_cost'] for r in rs]),'mean_gated_spend':np.mean([r['gated_cost'] for r in rs]),'switch_rate':np.mean([r['audit_switch'] for r in rs])}
        for m in ('default_hits','alternate_hits','interleave_hits','audit_hits','gated_hits'):
            d[m]=np.mean([r[m] for r in rs])
            vals=np.array([np.mean([r[m]-r['default_hits'] for r in rs if r['split']==s]) for s in range(10)])
            boot=vals[rng.integers(0,10,(20000,10))].mean(1)
            d[m+'_minus_default']=float(vals.mean());d[m+'_ci95_low']=float(np.quantile(boot,.025));d[m+'_ci95_high']=float(np.quantile(boot,.975))
        summary.append(d)
    for name,data in [('decision_replay.csv',rows),('decision_summary.csv',summary)]:
        with (OUT/name).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
