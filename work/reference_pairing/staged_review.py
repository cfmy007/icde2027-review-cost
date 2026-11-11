from pathlib import Path
import csv,json,os
from collections import defaultdict
import numpy as np
from decision_replay import audit,interleave

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
P=json.loads(Path(os.environ.get('STAGED_PROTOCOL',HERE/'staged_protocol.json')).read_text(encoding='utf8'))
OUT=Path(os.environ.get('STAGED_OUTPUT',HERE));OUT.mkdir(parents=True,exist_ok=True)

def cases():
    root=WORK/'ssh_followup/audit_frontier_result'
    for name in ('c10','c100','dyna','nli','nli-holdout'):
        with np.load(root/(name+'.panels.npz')) as z:
            for i,g in enumerate(z['group']):
                r=z[f'order{i}'].astype(int)
                yield name,i,int(g),r[0],r[1],z[f'error{i}'].astype(bool)
    root=WORK/'strong_eab_evidence'
    for name,folder in [('fashion-RN18','fashion/RN18'),('fashion-EB0','fashion/EB0'),('allenoise','allenoise')]:
        for i in range(10):
            path=root/folder/str(i)
            with (path/'choices.csv').open(newline='') as f:
                c={(r['source'],r['source_purity']):r['selected'] for r in csv.DictReader(f) if r['target_purity']=='human'}
            with np.load(path/('data.npz' if name=='allenoise' else 'queues.npz')) as z:
                target=z['target'];e=(z['truth'][target]!=z['claims'][target]);a=z['human/'+c['self-10','clean']];b=z['human/'+c['external-10','human']]
                yield name,i,i,a,b,e
    root=WORK/'ssh_followup/blind_rate_release_check/results/located'
    for name in ('ucc','wiki','go','d3'):
        with np.load(root/name/'full.plan.npz') as p,np.load(root/name/'full.judge.npz') as j:
            names=p['arms'].tolist()
            yield name,0,0,p['orders'][names.index('budget')],p['orders'][names.index('auc')],j['errors'][0]

def prefix_inputs(a,b,n):
    k=len(a);assert len(b)==k and len(np.unique(a))==len(np.unique(b))==k
    cat=np.full(n,-1,dtype=np.int8);cat[a]=0;cat[b]=1
    shared=np.intersect1d(a,b,assume_unique=True);cat[shared]=2
    strata=[np.flatnonzero(cat==j) for j in range(3)]
    assert len(strata[0])==len(strata[1])
    return cat,strata,(a,b,interleave(a,b))

def sample(strata,requested,rng,excluded):
    chosen=[]
    for ids,m in zip(strata,requested):
        remain=ids[~excluded[ids]];m=min(m,len(remain))
        if m:chosen.extend(rng.choice(remain,m,replace=False).tolist())
    return np.array(chosen,dtype=int)

def posterior(obs,values,cat):
    x=np.bincount(cat[obs],minlength=3);y=np.bincount(cat[obs],weights=values,minlength=3)
    return 1+y,1+x-y

def expect(options,obs,values,cat,k):
    alpha,beta=posterior(obs,values,cat);rate=alpha/(alpha+beta)
    seen=np.zeros(len(cat),dtype=bool);seen[obs]=True
    remain=k-len(obs);assert remain>=0
    # Paid hits count regardless of final candidate. Remaining records use the chosen queue.
    est=[]
    for q in options:
        extra=q[~seen[q]][:remain];assert len(extra)==remain
        count=np.bincount(cat[extra],minlength=3)
        est.append(float(values.sum()+count@rate))
    return np.array(est),alpha,beta

def observe(ids,truth):
    return truth[ids].astype(float)

def run_case(a,b,truth,rng):
    k=len(a);cat,strata,options=prefix_inputs(a,b,len(truth));m=int(.02*k)
    empty=np.zeros(len(truth),bool)
    pilot=sample(strata,(m,m,m),rng,empty);pilotvals=observe(pilot,truth)
    seen=empty.copy();seen[pilot]=True
    stop,alpha,beta=expect(options,pilot,pilotvals,cat,k);stop_choice=int(stop.argmax());stop_value=float(stop.max())
    trial=[]
    for _ in range(P['simulation']['posterior_draws']):
        new=sample(strata,(m,m,0),rng,seen)
        rates=rng.beta(alpha,beta);v=(rng.random(len(new))<rates[cat[new]]).astype(float)
        estimate,_,_=expect(options,np.r_[pilot,new],np.r_[pilotvals,v],cat,k)
        trial.append(float(estimate.max()))
    predicted_gain=float(np.mean(trial)-stop_value)
    remaining=sum(int((~seen[ids]).sum()) for ids in strata[:2])
    cont=predicted_gain>0 and len(pilot)+2*m<=k and m>0 and remaining>0
    candidate_more=sample(strata,(m,m,0),rng,seen) if remaining and m else np.array([],dtype=int)
    cfobs=np.r_[pilot,candidate_more];cfvals=np.r_[pilotvals,observe(candidate_more,truth)]
    cfscore,_,_=expect(options,cfobs,cfvals,cat,k);cfchoice=int(cfscore.argmax())
    more=candidate_more if cont else np.array([],dtype=int)
    obs=np.r_[pilot,more];vals=np.r_[pilotvals,observe(more,truth)]
    estimate,_,_=expect(options,obs,vals,cat,k);choice=int(estimate.argmax())
    def realized(obs,choice):
        s=np.zeros(len(truth),dtype=bool);s[obs]=True
        extra=options[choice][~s[options[choice]]][:k-len(obs)]
        assert len(np.unique(np.r_[obs,extra]))==k
        return int(truth[obs].sum()+truth[extra].sum())
    fixed,spent,switched=audit(a,b,truth,rng,False)
    return dict(pilot_n=len(pilot),second_n=len(more),continued=int(cont),predicted_gain=predicted_gain,
        pilot_rate_a=float(alpha[0]/(alpha[0]+beta[0])),pilot_rate_b=float(alpha[1]/(alpha[1]+beta[1])),
        pilot_rate_common=float(alpha[2]/(alpha[2]+beta[2])),counterfactual_second_n=len(candidate_more),
        counterfactual_cont_hits=realized(cfobs,cfchoice),
        stage_choice=choice,stop_choice=stop_choice,staged_hits=realized(obs,choice),
        pilot_only_hits=realized(pilot,stop_choice),default_hits=int(truth[a].sum()),
        alt_hits=int(truth[b].sum()),mix_hits=int(truth[options[2]].sum()),fixed_audit_hits=fixed,fixed_audit_n=spent)

def main():
    rows=[];rng=np.random.default_rng(P['simulation']['seed'])
    for name,panel,group,a,b,e in cases():
        for rep in range(P['simulation']['pilot_repetitions_per_panel']):
            r=run_case(a,b,e,rng)
            rows.append(dict(dataset=name,panel=panel,group=group,rep=rep,budget=len(a),exclusive=len(np.setdiff1d(a,b)),**r))
        print(name,panel,flush=True)
    with (OUT/'staged_rows.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups=defaultdict(list)
    for r in rows:groups[r['dataset']].append(r)
    out=[]; bootrng=np.random.default_rng(227900)
    for name,rs in groups.items():
        ids=sorted({r['group'] for r in rs});s=dict(dataset=name,panels=len({r['panel'] for r in rs}),groups=len(ids),budget=rs[0]['budget'],
            mean_pilot=np.mean([r['pilot_n'] for r in rs]),mean_second=np.mean([r['second_n'] for r in rs]),
            continue_rate=np.mean([r['continued'] for r in rs]))
        for key in ('staged_hits','pilot_only_hits','default_hits','alt_hits','mix_hits','fixed_audit_hits'):
            s[key]=float(np.mean([r[key] for r in rs]))
        for key in ('pilot_only_hits','default_hits','mix_hits','fixed_audit_hits'):
            diff=np.array([np.mean([r['staged_hits']-r[key] for r in rs if r['group']==g]) for g in ids])
            s['stage_minus_'+key]=float(diff.mean())
            if len(ids)>1:
                boot=diff[bootrng.integers(0,len(ids),(20000,len(ids)))].mean(1)
                s['stage_minus_'+key+'_low'],s['stage_minus_'+key+'_high']=map(float,np.quantile(boot,[.025,.975]))
            else:s['stage_minus_'+key+'_low']=s['stage_minus_'+key+'_high']=''
        out.append(s)
    with (OUT/'staged_summary.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
