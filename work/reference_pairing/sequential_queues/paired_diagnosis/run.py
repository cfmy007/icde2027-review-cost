from pathlib import Path
import csv,sys,importlib.util,numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prior',HERE.parent/'paired_displacement'/'run.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
eq=prior.eq;extra=prior.extra

def choice_samples(ids,s,ranked,rng):
 if not ranked:return rng.choice(ids,s,replace=False)
 parts=np.array_split(ids,2);sizes=[min((s+1)//2,len(parts[0])),0];sizes[1]=min(s-sizes[0],len(parts[1]));sizes[0]=min(s-sizes[1],len(parts[0]))
 return np.concatenate([np.empty(0,dtype=int),*(rng.choice(p,size=k,replace=False) for p,k in zip(parts,sizes) if k)])

def run(a,b,e,seed,fraction,ranked):
 B=len(a);n=len(e);ina=np.zeros(n,bool);ina[a]=True;x=b[~ina[b]];m=min(len(x),max(1,int(.2*B)))
 if not m:
  return dict(m=0,s=0,pred=0.,actual=0,raw=0.,full=0.,select=0,q=a,q0=a,pilot=np.array([],int))
 x=x[:m];tail=a[B-m:];s=min(m,max(1,int(fraction*B)));rng=np.random.default_rng(seed)
 xp=choice_samples(x,s,ranked,rng);tp=choice_samples(tail,s,ranked,rng);pilot=np.r_[xp,tp]
 q0=prior.complete(pilot,a,B,n);q1=prior.complete(pilot,np.r_[a[:B-m],x],B,n)
 mask0=np.zeros(n,bool);mask1=np.zeros(n,bool);mask0[q0]=True;mask1[q1]=True;add=q1[~mask0[q1]];remove=q0[~mask1[q0]]
 components=[x,tail] if not ranked else [*np.array_split(x,2),*np.array_split(tail,2)]
 samples=[xp,tp] if not ranked else [xp[np.isin(xp,v)] for v in np.array_split(x,2)]+[tp[np.isin(tp,v)] for v in np.array_split(tail,2)]
 coefficients=[];posterior=[];raw=[];full=[]
 for g,p in zip(components,samples):
  mark=np.zeros(n,bool);mark[g]=True;coefficients.append(int(mark[add].sum()-mark[remove].sum()))
  posterior.append((1+int(e[p].sum()))/(len(p)+2));raw.append(float(e[p].mean()) if len(p) else .5);full.append(float(e[g].mean()) if len(g) else .5)
 coeff=np.array(coefficients);pred=float(coeff@posterior);rawpred=float(coeff@raw);fullpred=float(coeff@full);actual=int(e[q1].sum()-e[q0].sum());selected=int(pred>0)
 assert abs((pred-rawpred)+(rawpred-fullpred)+(fullpred-actual)-(pred-actual))<1e-9
 return dict(m=m,s=s,pred=pred,raw=rawpred,full=fullpred,actual=actual,select=selected,q=q1 if selected else q0,q0=q0,pilot=pilot)

def main():
 jobs=[(n,eq.new.ROOT/('heldout_pico' if n=='heldout_pico' else 'new_crowd/'+n),False) for n in eq.new.CATALOG+('heldout_pico',)]
 jobs += [(n,eq.hist.ROOT/n,False) for n in eq.hist.TARGETS]
 jobs += [(n,extra.ROOT/loc,ext) for n,loc,ext in extra.JOBS]
 priorrows={}
 with (HERE.parent/'paired_displacement'/'rows.csv').open(encoding='utf8',newline='') as f:
  for r in csv.DictReader(f):
   if r['dataset'] in ('SBIC','NoisyAG-News'):priorrows[r['dataset'],r['seed'],int(r['rep'])]=r
 rows=[]
 for job,(name,path,external) in enumerate(jobs):
  if name not in ('SBIC','NoisyAG-News'):continue
  with np.load(path/('scores.npz' if external else 'target.npz')) as z:
   ranks=z['target_rank' if external else 'rank'].astype(int);e=z['target_error' if external else 'errors'].astype(bool);names=z['names'].astype(str).tolist()
  with (path/'paired.csv').open(encoding='utf8',newline='') as f: panels=[p for p in csv.DictReader(f) if float(p['budget'])==.1 and (external or p['menu']=='expanded')]
  B=round(.1*len(e));assert len(panels)==10
  for pi,p in enumerate(panels):
   ia=names.index(p['random']) if external else int(p['random']);ib=names.index(p['difficulty']) if external else int(p['difficulty']);a=ranks[ia][:B];b=ranks[ib][:B]
   mix=eq.interleave(a,b);fk=eq.fixed(a,b,e);cap,_=eq.loss.run(a,b,e);base=[int(e[v].sum()) for v in (a,mix,fk,cap)]
   for rep in range(30):
    seed=78562000+100000*job+1000*pi+rep
    for fraction in (.02,.04,.08):
     for ranked in (False,True):
      z=run(a,b,e,seed,fraction,ranked);q=z['q'];stop=z['q0'];assert len(q)==len(set(q))==B
      if fraction==.02 and not ranked:
       previous=priorrows[name,p['seed'],rep]
       assert abs(z['pred']-float(previous['predicted_gain']))<1e-9 and z['actual']==int(previous['actual_increment']) and z['select']==int(previous['select_alt']) and int(e[q].sum())==int(previous['paired'])
      if rep==0:
       altered=e.copy();unseen=np.ones(len(e),bool);unseen[q]=False;altered[unseen]=~altered[unseen];z2=run(a,b,altered,seed,fraction,ranked)
       assert np.array_equal(z2['q'],q) and z2['select']==z['select'] and abs(z2['pred']-z['pred'])<1e-9
      rows.append(dict(dataset=name,source_seed=p['seed'],rep=rep,B=B,m=z['m'],per_side=fraction,ranked=int(ranked),pilot_n=len(z['pilot']),selected=z['select'],predicted=z['pred'],actual_increment=z['actual'],smoothing=z['pred']-z['raw'],sampling=z['raw']-z['full'],position=z['full']-z['actual'],default=base[0],interleave=base[1],fixed_k=base[2],feedback=base[3],stop=int(e[stop].sum()),final=int(e[q].sum())))
  print(name,'completed',flush=True)
 with (HERE/'rows.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 out=[]
 for name in ('SBIC','NoisyAG-News'):
  for frac in (.02,.04,.08):
   for ranked in (0,1):
    rs=[r for r in rows if r['dataset']==name and r['per_side']==frac and r['ranked']==ranked];d=dict(dataset=name,per_side=frac,ranked=ranked,n=len(rs))
    for key in ('pilot_n','selected','predicted','actual_increment','smoothing','sampling','position','default','interleave','fixed_k','feedback','stop','final'):
     d[key]=float(np.mean([r[key] for r in rs]))
    d['mae']=float(np.mean([abs(r['predicted']-r['actual_increment']) for r in rs]))
    selected=[r for r in rs if r['selected']];d['selected_n']=len(selected)
    for key in ('predicted','actual_increment','smoothing','sampling','position'):
     d['selected_'+key]=float(np.mean([r[key] for r in selected])) if selected else ''
    out.append(d)
 with (HERE/'summary.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
if __name__=='__main__':main()
