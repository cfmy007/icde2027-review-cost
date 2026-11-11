from pathlib import Path
from functools import lru_cache
import importlib.util,csv,numpy as np
from scipy.special import betainc,betaln
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prior',HERE.parent/'paired_displacement'/'run.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
eq=prior.eq;extra=prior.extra
nodes,weights=np.polynomial.legendre.leggauss(128);nodes=(nodes+1)/2;weights=weights/2
@lru_cache(None)
def prob_superior(xhits,thits,s):
 ax,bx=xhits+1,s-xhits+1;at,bt=thits+1,s-thits+1
 density=np.exp((ax-1)*np.log(nodes)+(bx-1)*np.log1p(-nodes)-betaln(ax,bx))
 return float(np.sum(weights*density*betainc(at,bt,nodes)))

def run(a,b,e,seed):
 B=len(a);n=len(e);ina=np.zeros(n,bool);ina[a]=True;x=b[~ina[b]];m=min(len(x),max(1,int(.2*B)))
 if m==0:
  return dict(m=0,s=0,pilot_n=0,prob=.5,chosen=dict(original=0,confidence=0,confirmation=0),q0=a,q1=a,pred=0.)
 x=x[:m];tail=a[B-m:];s=min(m,max(1,int(.02*B)));rng=np.random.default_rng(seed)
 xp=rng.choice(x,size=s,replace=False);tp=rng.choice(tail,size=s,replace=False);pilot=np.r_[xp,tp]
 q0=prior.complete(pilot,a,B,n);q1=prior.complete(pilot,np.r_[a[:B-m],x],B,n)
 mark0=np.zeros(n,bool);mark1=np.zeros(n,bool);mark0[q0]=True;mark1[q1]=True;add=q1[~mark0[q1]];remove=q0[~mark1[q0]]
 markx=np.zeros(n,bool);markt=np.zeros(n,bool);markx[x]=True;markt[tail]=True
 px=(1+int(e[xp].sum()))/(s+2);pt=(1+int(e[tp].sum()))/(s+2)
 pred=(int(markx[add].sum())-int(markx[remove].sum()))*px+(int(markt[add].sum())-int(markt[remove].sum()))*pt
 p=prob_superior(int(e[xp].sum()),int(e[tp].sum()),s);split=s//2
 confirmation=bool(split>0 and len(add)>0 and float(e[xp[:split]].mean())>float(e[tp[:split]].mean()) and float(e[xp[split:]].mean())>float(e[tp[split:]].mean()))
 return dict(m=m,s=s,pilot_n=len(pilot),prob=p,chosen=dict(original=int(pred>0),confidence=int(len(add)>0 and p>.95),confirmation=int(confirmation)),q0=q0,q1=q1,pred=float(pred))

def main():
 jobs=[(n,eq.new.ROOT/('heldout_pico' if n=='heldout_pico' else 'new_crowd/'+n),False) for n in eq.new.CATALOG+('heldout_pico',)]
 jobs += [(n,eq.hist.ROOT/n,False) for n in eq.hist.TARGETS]
 jobs += [(n,extra.ROOT/loc,ext) for n,loc,ext in extra.JOBS]
 previous={}
 with (HERE.parent/'paired_displacement'/'rows.csv').open(encoding='utf8',newline='') as f:
  for r in csv.DictReader(f):previous[r['dataset'],r['seed'],int(r['rep'])]=r
 rows=[];summary=[]
 for job,(name,path,external) in enumerate(jobs):
  with np.load(path/('scores.npz' if external else 'target.npz'),allow_pickle=False) as z:
   ranks=z['target_rank' if external else 'rank'].astype(int);e=z['target_error' if external else 'errors'].astype(bool);names=z['names'].astype(str).tolist()
  with (path/'paired.csv').open(newline='',encoding='utf8') as f:panels=[p for p in csv.DictReader(f) if float(p['budget'])==.1 and (external or p['menu']=='expanded')]
  assert len(panels)==10
  B=round(.1*len(e));start=len(rows)
  for pi,p in enumerate(panels):
   ia=names.index(p['random']) if external else int(p['random']);ib=names.index(p['difficulty']) if external else int(p['difficulty']);a=ranks[ia][:B];b=ranks[ib][:B]
   for rep in range(30):
    seed=78562000+100000*job+1000*pi+rep;z=run(a,b,e,seed);pr=previous[name,p['seed'],rep]
    assert z['chosen']['original']==int(pr['select_alt']) and abs(z['pred']-float(pr['predicted_gain']))<1e-9
    q0=z['q0'];q1=z['q1'];h0=int(e[q0].sum());h1=int(e[q1].sum());original=h1 if z['chosen']['original'] else h0
    assert original==int(pr['paired']) and h0==int(pr['paired_stop'])
    if rep==0:
     for pol,take in z['chosen'].items():
      q=q1 if take else q0;assert len(q)==len(set(q))==B
      modified=e.copy();unseen=np.ones(len(e),bool);unseen[q]=False;modified[unseen]=~modified[unseen];z2=run(a,b,modified,seed)
      assert z2['chosen'][pol]==take and np.array_equal((z2['q1'] if take else z2['q0']),q)
    rows.append(dict(dataset=name,seed=p['seed'],rep=rep,B=B,pilot_n=z['pilot_n'],prob=z['prob'],pred=z['pred'],actual_increment=h1-h0,
      default=int(pr['default']),interleave=int(pr['interleave']),fixed_k=int(pr['fixed_k']),feedback=int(pr['feedback']),stop=h0,original=original,
      confidence=h1 if z['chosen']['confidence'] else h0,confirmation=h1 if z['chosen']['confirmation'] else h0,
      original_take=z['chosen']['original'],confidence_take=z['chosen']['confidence'],confirmation_take=z['chosen']['confirmation']))
  rs=rows[start:];d=dict(dataset=name,B=B,panels=10,reps=30)
  for key in ('pilot_n','default','interleave','fixed_k','feedback','stop','original','confidence','confirmation','original_take','confidence_take','confirmation_take'):
   d[key]=float(np.mean([r[key] for r in rs]))
  for pol in ('original','confidence','confirmation'):
   selected=[r for r in rs if r[pol+'_take']];d[pol+'_selected_n']=len(selected)
   d[pol+'_selected_pred']=float(np.mean([r['pred'] for r in selected])) if selected else ''
   d[pol+'_selected_actual']=float(np.mean([r['actual_increment'] for r in selected])) if selected else ''
   d[pol+'_vs_original']=float(np.mean([r[pol]-r['original'] for r in rs]))
   d[pol+'_vs_stop']=float(np.mean([r[pol]-r['stop'] for r in rs]))
  summary.append(d);print(name,'direct,mix,orig,conf,confirm',*[round(d[k],2) for k in ('default','interleave','original','confidence','confirmation')],flush=True)
 for filename,data in [('rows.csv',rows),('summary.csv',summary)]:
  with (HERE/filename).open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
if __name__=='__main__':main()
