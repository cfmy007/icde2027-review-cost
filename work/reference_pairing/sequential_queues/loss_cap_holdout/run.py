from pathlib import Path
from collections import defaultdict
import importlib.util,csv,sys,numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cap',HERE.parent/'new_target_test'/'loss_cap.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
from decision_replay import interleave
ROOT=HERE.parents[2]/'decision_closure_20261111'/'inputs'/'historical'
TARGETS=('AudioSet','Davidson','HateXplain','MBIC','Measuring-Hate-Speech','SBIC','Stanford-Politeness')

def main():
 rows=[];summary=[];rng=np.random.default_rng(991738)
 for name in TARGETS:
  folder=ROOT/name
  with np.load(folder/'target.npz') as z:r=z['rank'].astype(int);e=z['errors'].astype(bool);_,groups=np.unique(z['group'],return_inverse=True)
  with (folder/'paired.csv').open(newline='',encoding='utf8') as f:panels=[x for x in csv.DictReader(f) if x['menu']=='expanded' and float(x['budget'])==.1]
  assert len(panels)==10 and len({x['seed'] for x in panels})==10
  B=round(.1*len(e));ng=groups.max()+1;cont=[]
  for p in panels:
   a=r[int(p['random'])][:B];b=r[int(p['difficulty'])][:B];q,bad=mod.run(a,b,e);m=interleave(a,b);delta=int(e[a].sum())-int(e[q].sum());k=int(.05*B)
   altered=e.copy();mask=np.ones(len(e),bool);mask[q]=False;altered[mask]=~altered[mask]
   assert np.array_equal(q,mod.run(a,b,altered)[0]);assert len(q)==len(np.unique(q))==B and delta<=bad<=k
   cont.append(np.array([np.bincount(groups[x[e[x]]],minlength=ng) for x in (a,b,m,q)]))
   rows.append(dict(dataset=name,seed=p['seed'],N=len(e),B=B,cap=k,source_random=p['random'],source_difficulty=p['difficulty'],same_queue=int(p['random']==p['difficulty']),outside=len(set(q)-set(a)),outside_no_disagreement=bad,default=int(e[a].sum()),alternative=int(e[b].sum()),interleave=int(e[m].sum()),capped=int(e[q].sum()),shortfall=delta))
  vals=np.stack(cont);mean=vals.sum(axis=2).mean(axis=0);z=[x for x in rows if x['dataset']==name]
  out=dict(dataset=name,N=len(e),B=B,groups=ng,panels=len(panels),cap=int(.05*B),same_queue=sum(x['same_queue'] for x in z),outside=np.mean([x['outside'] for x in z]),outside_no_disagreement=np.mean([x['outside_no_disagreement'] for x in z]),default=mean[0],alternative=mean[1],interleave=mean[2],capped=mean[3],vs_default=mean[3]-mean[0],vs_interleave=mean[3]-mean[2],max_observed_shortfall=max(x['shortfall'] for x in z))
  if ng>1:
   w=rng.multinomial(ng,np.full(ng,1/ng),size=2000)
   for j,key in ((0,'default'),(2,'interleave')):
    v=(vals[:,3,:]-vals[:,j,:]).mean(axis=0);out['vs_'+key+'_low'],out['vs_'+key+'_high']=np.quantile(w@v,[.025,.975])
  else:
   for key in ('default','interleave'):out['vs_'+key+'_low']=out['vs_'+key+'_high']=''
  summary.append(out);print(name,'B',B,'default/mix/cap',mean[[0,2,3]],'outside',out['outside'],flush=True)
 for filename,data in [('rows.csv',rows),('summary.csv',summary)]:
  with (HERE/filename).open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
if __name__=='__main__':main()
