from pathlib import Path
import csv,json,importlib.util,numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('risk',HERE.parent/'equal_risk_comparison'/'run.py');risk=importlib.util.module_from_spec(spec);spec.loader.exec_module(risk)
spec=importlib.util.spec_from_file_location('extra',HERE.parent/'equal_risk_extra'/'run.py');extra=importlib.util.module_from_spec(spec);spec.loader.exec_module(extra)

def main():
 eq=risk.eq if hasattr(risk,'eq') else None
 new=risk.new;hist=risk.hist
 jobs=[(n,new.ROOT/('heldout_pico' if n=='heldout_pico' else 'new_crowd/'+n),False) for n in new.CATALOG+('heldout_pico',)]
 jobs +=[(n,hist.ROOT/n,False) for n in hist.TARGETS]
 jobs +=[(n,extra.ROOT/loc,ext) for n,loc,ext in extra.JOBS]
 rows=[];summary=[]
 for name,path,external in jobs:
  with np.load(path/('scores.npz' if external else 'target.npz'),allow_pickle=False) as z:
   ranks=z['target_rank' if external else 'rank'].astype(int);e=z['target_error' if external else 'errors'].astype(bool);names=z['names'].astype(str).tolist()
  with (path/'paired.csv').open(newline='',encoding='utf8') as f:panels=[p for p in csv.DictReader(f) if float(p['budget'])==.1 and (external or p['menu']=='expanded')]
  assert len(panels)==10; B=round(.1*len(e));k=int(.05*B);start=len(rows)
  for p in panels:
   ia=names.index(p['random']) if external else int(p['random']);ib=names.index(p['difficulty']) if external else int(p['difficulty']);a=ranks[ia][:B];b=ranks[ib][:B]
   membership=np.zeros(len(e),bool);membership[a]=True;outside=b[~membership[b]];d=len(outside);assert d==int((~np.isin(a,b)).sum())
   default=int(e[a].sum());alternative=int(e[b].sum());gap=abs(default-alternative);assert gap<=d
   feedback,f=risk.loss.run(a,b,e);fixed=risk.fixed(a,b,e);inter=risk.interleave(a,b);yh=int(e[feedback].sum());yf=int(e[fixed].sum());ym=int(e[inter].sum());signed=default-yh;actual=max(0,signed)
   assert len(feedback)==len(set(feedback))==B and signed<=f<=k and default-yf<=k and actual<=f
   nout=int((~membership[feedback]).sum())
   rows.append(dict(dataset=name,seed=p['seed'],N=len(e),B=B,k=k,source_random=ia,source_difficulty=ib,d=d,d_per_B=d/B,default=default,alternative=alternative,absolute_gap=gap,gap_per_B=gap/B,slack=d-gap,tightness=(gap/d if d else ''),feedback=yh,fixed_k=yf,interleave=ym,outside_reviews=nout,observed_failures=f,signed_shortfall=signed,loss=actual,cap_slack=k-actual,observed_bound_slack=f-actual,explores=int(nout>0),cap_binding=int(k>0 and f==k)))
  rs=rows[start:];nonzero=[x for x in rs if x['d']>0];active=[x for x in rs if x['outside_reviews']>0]
  v=dict(dataset=name,N=len(e),B=B,k=k,panels=10,zero_d=sum(x['d']==0 for x in rs),positive_k=int(k>0),exploring_panels=len(active))
  for field in ('d','d_per_B','absolute_gap','gap_per_B','slack','default','alternative','interleave','feedback','fixed_k','outside_reviews','observed_failures','signed_shortfall','loss','cap_slack','observed_bound_slack','cap_binding'):
   v['mean_'+field]=float(np.mean([x[field] for x in rs]))
  v['mean_tightness_dpositive']=float(np.mean([x['tightness'] for x in nonzero])) if nonzero else ''
  v['median_tightness_dpositive']=float(np.median([x['tightness'] for x in nonzero])) if nonzero else ''
  v['fraction_d_le_1pct']=float(np.mean([x['d']<=.01*x['B'] for x in rs]))
  v['fraction_d_le_5pct']=float(np.mean([x['d']<=.05*x['B'] for x in rs]))
  v['fraction_d_le_10pct']=float(np.mean([x['d']<=.10*x['B'] for x in rs]))
  summary.append(v)
 for filename,data in [('rows.csv',rows),('summary.csv',summary)]:
  with (HERE/filename).open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
 aggregate={}
 for threshold in (.01,.05,.1):
  aggregate[f'within_{int(100*threshold)}pct']={'panels':sum(r['d_per_B']<=threshold for r in rows),'total_panels':len(rows),'target_macro_fraction':float(np.mean([v[f'fraction_d_le_{int(100*threshold)}pct'] for v in summary])),'all_seeds_targets':sum(v[f'fraction_d_le_{int(100*threshold)}pct']==1 for v in summary)}
 nz=[r for r in rows if r['d']>0];active=[r for r in rows if r['explores']];kpositive=[r for r in rows if r['k']>0]
 aggregate.update(dict(targets=len(summary),panels=len(rows),zero_d_panels=sum(r['d']==0 for r in rows),all_zero_d_targets=sum(v['zero_d']==10 for v in summary),zero_k_panels=len(rows)-len(kpositive),nonzero_d_panels=len(nz),dpositive_tightness_median=float(np.median([r['tightness'] for r in nz])),dpositive_tightness_mean=float(np.mean([r['tightness'] for r in nz])),active_panels=len(active),active_mean_cap=float(np.mean([r['k'] for r in active])),active_mean_f=float(np.mean([r['observed_failures'] for r in active])),active_mean_actual_loss=float(np.mean([r['loss'] for r in active])),active_cap_binding=sum(r['cap_binding'] for r in active),active_gain=sum(r['signed_shortfall']<0 for r in active),active_tie=sum(r['signed_shortfall']==0 for r in active),active_loss=sum(r['signed_shortfall']>0 for r in active)))
 (HERE/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n',encoding='utf8');print(json.dumps(aggregate,indent=2),flush=True)
 for v in summary:print(v['dataset'],'B',v['B'],'d',round(v['mean_d'],1),'gap',round(v['mean_absolute_gap'],1),'ratio',v['mean_tightness_dpositive'],'cap',v['k'],'f',round(v['mean_observed_failures'],1),'loss',round(v['mean_loss'],1),'outside',round(v['mean_outside_reviews'],1),flush=True)
if __name__=='__main__':main()
