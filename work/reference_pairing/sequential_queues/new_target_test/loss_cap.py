from pathlib import Path
import csv,sys,numpy as np
from collections import defaultdict
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parent.parent))
import importlib.util
spec=importlib.util.spec_from_file_location('new_target_run',HERE/'run.py')
target_run=importlib.util.module_from_spec(spec);spec.loader.exec_module(target_run)
CATALOG,ROOT=target_run.CATALOG,target_run.ROOT
from switching import run as switch
from decision_replay import interleave

def run(a,b,e):
 B=len(a);limit=int(.05*B);bad=0;outside=[];aset=set(a.tolist())
 if limit:
  for x in b:
   if int(x) in aset:continue
   outside.append(int(x));bad+=int(not e[x])
   if bad==limit:break
 q=np.r_[a[:B-len(outside)],np.array(outside,dtype=int)]
 return q,bad

def main():
 rows=[]
 for name in CATALOG+('heldout_pico',):
  folder=ROOT/('heldout_pico' if name=='heldout_pico' else 'new_crowd/'+name)
  with np.load(folder/'target.npz') as z:r=z['rank'].astype(int);e=z['errors'].astype(bool)
  with (folder/'paired.csv').open(encoding='utf8',newline='') as f:panels=[x for x in csv.DictReader(f) if x['menu']=='expanded' and float(x['budget'])==.1]
  B=round(.1*len(e))
  for p in panels:
   a,b=r[int(p['random'])][:B],r[int(p['difficulty'])][:B];q,bad=run(a,b,e);m=interleave(a,b);s,_,_=switch(a,b,e,.05,True)
   altered=e.copy();mask=np.ones(len(e),bool);mask[q]=False;altered[mask]=~altered[mask];assert np.array_equal(q,run(a,b,altered)[0]); assert len(q)==len(np.unique(q))==B and int(e[a].sum())-int(e[q].sum())<=bad<=int(.05*B)
   rows.append(dict(dataset=name,seed=p['seed'],B=B,limit=int(.05*B),bad=bad,outside=len(set(q)-set(a)),default=int(e[a].sum()),mix=int(e[m].sum()),switch=s,capped=int(e[q].sum())))
 out=[];by=defaultdict(list)
 for r in rows:by[r['dataset']].append(r)
 for name,rs in by.items():
  d=dict(dataset=name,B=rs[0]['B'],limit=rs[0]['limit'])
  for k in ('bad','outside','default','mix','switch','capped'):d[k]=float(np.mean([x[k] for x in rs]))
  out.append(d);print(name,*[round(d[k],2) for k in ('default','mix','switch','capped','outside','bad')])
 for name,data in [('loss_cap_rows.csv',rows),('loss_cap_summary.csv',out)]:
  with (HERE/name).open('w',newline='',encoding='utf8') as f:
   w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
if __name__=='__main__':main()
