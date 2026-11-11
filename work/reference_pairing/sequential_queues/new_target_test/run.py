from pathlib import Path
from collections import defaultdict
import csv,sys,numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parent.parent))
from switching import run
from decision_replay import interleave
ROOT=HERE.parents[2]/'decision_closure_20261111'/'inputs'
CATALOG=('alia','amazon','authorship','coda','convabuse','crowdtruth_cause','crowdtruth_treat','dices','disagg_italian','gahd','germs','misogyny','mvsc','sentimp','snow_external_anger','snow_external_fear','snow_external_joy','varierr','web_nist')

def main():
 rows=[];summ=[];rng=np.random.default_rng(412611)
 for name in CATALOG+('heldout_pico',):
  path=ROOT/('heldout_pico' if name=='heldout_pico' else 'new_crowd/'+name)
  with np.load(path/'target.npz',allow_pickle=False) as z:
   rank=z['rank'].astype(int);e=z['errors'].astype(bool);rawgroup=z['group'] if 'group' in z else z['ids'];_,group=np.unique(rawgroup,return_inverse=True)
  with (path/'paired.csv').open(newline='',encoding='utf8') as f:
   panels=[r for r in csv.DictReader(f) if r['menu']=='expanded' and float(r['budget'])==.1]
  if len(panels)!=10 or len({p['seed'] for p in panels})!=10:raise ValueError((name,'expected ten source seeds',len(panels)))
  n=len(e);B=round(.1*n);contrib=[]
  for panel in panels:
   ia=int(panel['random']);ib=int(panel['difficulty']);a=rank[ia][:B];b=rank[ib][:B];m=interleave(a,b)
   hits,alloc,q=run(a,b,e,.05,trace=True);q=np.array(q,int)
   assert len(q)==B and len(np.unique(q))==B and hits==int(e[q].sum())
   altered=e.copy();unseen=np.ones(n,bool);unseen[q]=False;altered[unseen]=~altered[unseen]
   assert run(a,b,altered,.05,trace=True)[2]==q.tolist()
   counts=np.array([int(e[x].sum()) for x in (a,b,m,q)])
   c=np.array([np.bincount(group[x[e[x]]],minlength=group.max()+1) for x in (a,b,m,q)])
   contrib.append(c)
   rows.append(dict(dataset=name,seed=panel['seed'],n=n,budget=B,source_random_index=ia,source_difficulty_index=ib,
       same_candidate=int(ia==ib),queue_overlap=len(np.intersect1d(a,b)),default=int(counts[0]),alternative=int(counts[1]),mix=int(counts[2]),switch=int(counts[3]),from_default=alloc[0],from_alternative=alloc[1]))
  arr=np.array(contrib);point=arr.sum(axis=2).mean(axis=0);ng=arr.shape[2]
  out=dict(dataset=name,n=n,budget=B,groups=ng,panels=len(panels),same_candidate=sum(r['same_candidate'] for r in rows if r['dataset']==name),default=point[0],alternative=point[1],mix=point[2],switch=point[3],vs_default=point[3]-point[0],vs_mix=point[3]-point[2])
  if ng>1:
   weights=rng.multinomial(ng,np.full(ng,1/ng),size=5000)
   diff=(arr[:,3,:]-arr[:,0,:]).mean(axis=0);diffmix=(arr[:,3,:]-arr[:,2,:]).mean(axis=0)
   out['vs_default_low'],out['vs_default_high']=np.quantile(weights@diff,[.025,.975]);out['vs_mix_low'],out['vs_mix_high']=np.quantile(weights@diffmix,[.025,.975])
  else:out['vs_default_low']=out['vs_default_high']=out['vs_mix_low']=out['vs_mix_high']=''
  summ.append(out);print(name,'B',B,'default/mix/switch',point[[0,2,3]],flush=True)
 for file,data in [('rows.csv',rows),('summary.csv',summ)]:
  with (HERE/file).open('w',newline='',encoding='utf8') as f:
   w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
if __name__=='__main__':main()
