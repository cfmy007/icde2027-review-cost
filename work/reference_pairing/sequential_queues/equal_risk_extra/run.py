from pathlib import Path
import importlib.util,csv,sys,numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('equal_risk',HERE.parent/'equal_risk_comparison'/'run.py');eq=importlib.util.module_from_spec(spec);spec.loader.exec_module(eq)
ROOT=HERE.parents[2]/'decision_closure_20261111'/'inputs'
JOBS=(('NoisyAG-News','heldout/16-two-targets__results_ag',True),('PGR','heldout/16-two-targets__results_pgr',True),('nli_dnc','new_crowd/nli_dnc',False),('nli_mnli','new_crowd/nli_mnli',False),('nli_snli','new_crowd/nli_snli',False),('snow_rte','new_crowd/snow_rte',False))

def main():
 rows=[];summary=[];rng=np.random.default_rng(772911)
 for name,loc,external in JOBS:
  path=ROOT/loc
  with np.load(path/('scores.npz' if external else 'target.npz'),allow_pickle=False) as z:
   ranks=z['target_rank' if external else 'rank'].astype(int);err=z['target_error' if external else 'errors'].astype(bool);names=z['names'].astype(str).tolist();_,group=np.unique(z['group'] if 'group' in z else z['ids'],return_inverse=True)
  with (path/'paired.csv').open(newline='',encoding='utf8') as f:panels=[x for x in csv.DictReader(f) if float(x['budget'])==.1 and (external or x['menu']=='expanded')]
  assert len(panels)==10 and len({x['seed'] for x in panels})==10
  B=round(.1*len(err));k=int(.05*B);ng=int(group.max()+1);contrib=[]
  for p in panels:
   ia=names.index(p['random']) if external else int(p['random']);ib=names.index(p['difficulty']) if external else int(p['difficulty'])
   a=ranks[ia][:B];b=ranks[ib][:B];fixed=eq.fixed(a,b,err);feedback,bad=eq.loss.run(a,b,err);mix=eq.interleave(a,b)
   aa=set(a.tolist());of=[int(i) for i in fixed if int(i) not in aa];ob=[int(i) for i in feedback if int(i) not in aa]
   hit=[int(err[q].sum()) for q in (a,mix,fixed,feedback)];nf=sum(not err[i] for i in of)
   for q,policy in ((fixed,'fixed'),(feedback,'feedback')):
    assert len(q)==len(set(q))==B
    altered=err.copy();unseen=np.ones(len(err),bool);unseen[q]=False;altered[unseen]=~altered[unseen]
    q2=eq.fixed(a,b,altered) if policy=='fixed' else eq.loss.run(a,b,altered)[0]
    assert np.array_equal(q,q2)
   assert hit[0]-hit[2]<=k and hit[0]-hit[3]<=bad<=k and nf<=k
   contrib.append(np.array([np.bincount(group[q[err[q]]],minlength=ng) for q in (a,mix,fixed,feedback)]))
   rows.append(dict(dataset=name,seed=p['seed'],N=len(err),B=B,k=k,default_index=ia,alternative_index=ib,same_queue=int(ia==ib),outside_fixed=len(of),outside_feedback=len(ob),nonhit_fixed=nf,nonhit_feedback=bad,default=hit[0],interleave=hit[1],fixed=hit[2],feedback=hit[3],feedback_minus_fixed=hit[3]-hit[2]))
  arr=np.stack(contrib);means=arr.sum(axis=2).mean(axis=0);subset=[x for x in rows if x['dataset']==name]
  d=dict(dataset=name,N=len(err),B=B,k=k,groups=ng,panels=10,same_queue=sum(x['same_queue'] for x in subset),default=means[0],interleave=means[1],fixed=means[2],feedback=means[3],feedback_minus_fixed=means[3]-means[2],outside_fixed=np.mean([x['outside_fixed'] for x in subset]),outside_feedback=np.mean([x['outside_feedback'] for x in subset]),nonhit_fixed=np.mean([x['nonhit_fixed'] for x in subset]),nonhit_feedback=np.mean([x['nonhit_feedback'] for x in subset]))
  if ng>1:
   v=(arr[:,3,:]-arr[:,2,:]).mean(axis=0);samples=[]
   for _ in range(20):samples.extend(rng.multinomial(ng,np.full(ng,1/ng),size=100)@v)
   d['delta_low'],d['delta_high']=np.quantile(samples,[.025,.975])
  else:d['delta_low']=d['delta_high']=''
  summary.append(d);print(name,d,flush=True)
 for file,data in [('rows.csv',rows),('summary.csv',summary)]:
  with (HERE/file).open('w',encoding='utf8',newline='') as f:
   w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
if __name__=='__main__':main()
