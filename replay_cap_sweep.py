from pathlib import Path
import csv,importlib.util,numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
p=ROOT/'work/reference_pairing/sequential_queues/equal_risk_extra/run.py'
spec=importlib.util.spec_from_file_location('extra',p);extra=importlib.util.module_from_spec(spec);spec.loader.exec_module(extra);eq=extra.eq

def review(a,b,errors,budget,cap):
    pool=set(a.tolist());chosen=[];misses=0
    if cap:
        for record in b:
            if int(record) in pool:continue
            chosen.append(int(record));misses+=int(not errors[record])
            if misses==cap:break
    return np.r_[a[:budget-len(chosen)],np.asarray(chosen,dtype=int)]

def main():
    jobs=[(n,eq.new.ROOT/('heldout_pico' if n=='heldout_pico' else 'new_crowd/'+n)) for n in eq.new.CATALOG+('heldout_pico',)]
    jobs += [(n,eq.hist.ROOT/n) for n in eq.hist.TARGETS]
    jobs += [(n,extra.ROOT/loc) for n,loc,external in extra.JOBS if not external]
    fractions=(.01,.02,.05,.1,.2);records=[]
    for name,path in jobs:
        with np.load(path/'target.npz',allow_pickle=False) as data:ranks=data['rank'];errors=data['errors'].astype(bool)
        with (path/'paired.csv').open(encoding='utf8',newline='') as f:panels=[r for r in csv.DictReader(f) if r['menu']=='expanded' and float(r['budget'])==.1]
        B=round(.1*len(errors))
        for p in panels:
            a=ranks[int(p['random'])][:B];b=ranks[int(p['difficulty'])][:B];base=int(errors[a].sum())
            for fraction in fractions:
                cap=int(fraction*B);q=review(a,b,errors,B,cap)
                records.append(dict(dataset=name,seed=p['seed'],budget=B,cap_fraction=fraction,cap=cap,default=base,hits=int(errors[q].sum()),gain=int(errors[q].sum())-base))
    output=ROOT/'outputs';output.mkdir(exist_ok=True)
    with (output/'cap_sweep_rows.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    summary=[]
    for fraction in fractions:
        gains=np.array([np.mean([r['gain'] for r in records if r['dataset']==name and r['cap_fraction']==fraction]) for name,_ in jobs])
        summary.append(dict(cap_fraction=fraction,mean_gain=float(gains.mean()),better=int((gains>1e-9).sum()),equal=int((abs(gains)<=1e-9).sum()),worse=int((gains< -1e-9).sum())))
    with (output/'cap_sweep_summary.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
    x=np.arange(len(summary));y=[r['mean_gain'] for r in summary]
    fig,ax=plt.subplots(figsize=(4,2.5));ax.axhline(0,color='0.6',linewidth=.6);ax.plot(x,y,'o-',color='#0072b2',markersize=4)
    for i,r in enumerate(summary):
        label=f"{r['better']}/{r['equal']}/{r['worse']}";ax.annotate(label,(x[i],y[i]),xytext=(0,8 if i in (0,4) else -15),textcoords='offset points',ha='center',fontsize=7)
    ax.set_xticks(x,['1%','2%','5%','10%','20%']);ax.set_xlabel('default-loss cap $k$ as a share of budget $B$');ax.set_ylabel('gain over the default queue\n(corrected disagreements)');ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(output/'cap_sweep.pdf')
    expected=[(5,26,0),(4,22,5),(7,17,7),(12,7,12),(12,5,14)]
    assert [(r['better'],r['equal'],r['worse']) for r in summary]==expected
    assert abs(summary[-1]['mean_gain']+1.3226)<.001
    print('31 targets; cap sweep and Figure 4 values reproduced')
if __name__=='__main__':main()
