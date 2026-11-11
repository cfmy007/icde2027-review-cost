from pathlib import Path
import importlib.util,csv,sys,numpy as np
HERE=Path(__file__).resolve().parent

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

new=load('new_data',HERE.parent/'new_target_test'/'run.py')
hist=load('hist_data',HERE.parent/'loss_cap_holdout'/'run.py')
loss=load('loss_policy',HERE.parent/'new_target_test'/'loss_cap.py')
from decision_replay import interleave

def fixed(a,b,e):
 k=int(.05*len(a));aset=set(a.tolist());outside=[int(x) for x in b if int(x) not in aset][:k]
 return np.r_[a[:len(a)-len(outside)],np.array(outside,dtype=int)]

def main():
 jobs=[(n,new.ROOT/('heldout_pico' if n=='heldout_pico' else 'new_crowd/'+n)) for n in new.CATALOG+('heldout_pico',)]
 jobs += [(n,hist.ROOT/n) for n in hist.TARGETS]
 rows=[];summary=[];rng=np.random.default_rng(880711)
 for name,folder in jobs:
  with np.load(folder/'target.npz',allow_pickle=False) as z:
   ranks=z['rank'].astype(int);e=z['errors'].astype(bool);raw=z['group'] if 'group' in z else z['ids'];_,group=np.unique(raw,return_inverse=True)
  with (folder/'paired.csv').open(newline='',encoding='utf8') as f:panels=[x for x in csv.DictReader(f) if x['menu']=='expanded' and float(x['budget'])==.1]
  assert len(panels)==10 and len({x['seed'] for x in panels})==10
  B=round(.1*len(e));k=int(.05*B);ng=group.max()+1;contrib=[]
  for p in panels:
   a=ranks[int(p['random'])][:B];b=ranks[int(p['difficulty'])][:B];mix=interleave(a,b);fixedq=fixed(a,b,e);adaptive,bad=loss.run(a,b,e)
   assert len(fixedq)==len(adaptive)==B and len(np.unique(fixedq))==len(np.unique(adaptive))==B
   aset=set(a.tolist());fout=[int(x) for x in fixedq if int(x) not in aset];aout=[int(x) for x in adaptive if int(x) not in aset]
   fhits=int(e[fixedq].sum());ahits=int(e[adaptive].sum());dhits=int(e[a].sum());f_bad=sum(not e[i] for i in fout)
   assert len(fout)<=k and f_bad<=k and bad<=k and dhits-fhits<=k and dhits-ahits<=k
   for q,method in ((fixedq,'fixed'),(adaptive,'adaptive')):
    unseen=np.ones(len(e),bool);unseen[q]=False;modified=e.copy();modified[unseen]=~modified[unseen]
    other=fixed(a,b,modified) if method=='fixed' else loss.run(a,b,modified)[0]
    assert np.array_equal(other,q)
   contrib.append(np.array([np.bincount(group[x[e[x]]],minlength=ng) for x in (a,mix,fixedq,adaptive)]))
   rows.append(dict(dataset=name,seed=p['seed'],N=len(e),B=B,k=k,source_random=p['random'],source_difficulty=p['difficulty'],same_queue=int(p['random']==p['difficulty']),outside_fixed=len(fout),outside_feedback=len(aout),nonhit_fixed=f_bad,nonhit_feedback=bad,default=dhits,interleave=int(e[mix].sum()),fixed=fhits,feedback=ahits,feedback_minus_fixed=ahits-fhits,default_minus_fixed=dhits-fhits,default_minus_feedback=dhits-ahits))
  arr=np.stack(contrib);pts=arr.sum(axis=2).mean(axis=0);rr=[x for x in rows if x['dataset']==name];d=dict(dataset=name,N=len(e),B=B,k=k,groups=ng,panels=10,same_queue=sum(x['same_queue'] for x in rr),default=pts[0],interleave=pts[1],fixed=pts[2],feedback=pts[3],feedback_minus_fixed=pts[3]-pts[2],outside_fixed=np.mean([x['outside_fixed'] for x in rr]),outside_feedback=np.mean([x['outside_feedback'] for x in rr]),nonhit_fixed=np.mean([x['nonhit_fixed'] for x in rr]),nonhit_feedback=np.mean([x['nonhit_feedback'] for x in rr]),max_shortfall_fixed=max(x['default_minus_fixed'] for x in rr),max_shortfall_feedback=max(x['default_minus_feedback'] for x in rr))
  if ng>1:
   v=(arr[:,3,:]-arr[:,2,:]).mean(axis=0)
   vals=[]
   for _ in range(20):
    w=rng.multinomial(ng,np.full(ng,1/ng),size=100);vals.extend(w@v)
   d['delta_low'],d['delta_high']=np.quantile(vals,[.025,.975])
  else:d['delta_low']=d['delta_high']=''
  summary.append(d);print(name,'B',B,'k',k,'fixed',round(pts[2],2),'feedback',round(pts[3],2),'extra',round(d['outside_feedback']-d['outside_fixed'],2),flush=True)
 for filename,data in [('rows.csv',rows),('summary.csv',summary)]:
  with (HERE/filename).open('w',newline='',encoding='utf8') as f:
   writer=csv.DictWriter(f,fieldnames=data[0]);writer.writeheader();writer.writerows(data)
if __name__=='__main__':main()
