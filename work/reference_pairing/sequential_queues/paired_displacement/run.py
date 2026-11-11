from pathlib import Path
import sys,csv,importlib.util,numpy as np
HERE=Path(__file__).resolve().parent

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

eq=load('equalrisk',HERE.parent/'equal_risk_comparison'/'run.py')
extra=load('extra',HERE.parent/'equal_risk_extra'/'run.py')

def complete(pilot,order,B,n):
 seen=np.zeros(n,bool);seen[pilot]=True
 return np.r_[pilot,order[~seen[order]][:B-len(pilot)]]

def run(a,b,e,seed):
 B=len(a);n=len(e);in_a=np.zeros(n,bool);in_a[a]=True;x=b[~in_a[b]];m=min(len(x),max(1,int(.2*B)))
 if m==0:
  return dict(m=0,pilot_n=0,select_alt=0,predicted_gain=0.,actual_increment=0,review=a,stop=a,alt=a)
 x=x[:m];tail=a[B-m:];s=min(m,max(1,int(.02*B)))
 rng=np.random.default_rng(seed);xp=rng.choice(x,size=s,replace=False);tp=rng.choice(tail,size=s,replace=False);pilot=np.r_[xp,tp]
 q0=complete(pilot,a,B,n);candidate=np.r_[a[:B-m],x];q1=complete(pilot,candidate,B,n)
 mask0=np.zeros(n,bool);mask1=np.zeros(n,bool);mask0[q0]=True;mask1[q1]=True
 add=q1[~mask0[q1]];remove=q0[~mask1[q0]]
 maskx=np.zeros(n,bool);maskt=np.zeros(n,bool);maskx[x]=True;maskt[tail]=True
 assert np.all(maskx[add]|maskt[add]) and np.all(maskx[remove]|maskt[remove])
 px=(1+e[xp].sum())/(s+2);pt=(1+e[tp].sum())/(s+2)
 predicted=(int(maskx[add].sum())-int(maskx[remove].sum()))*px+(int(maskt[add].sum())-int(maskt[remove].sum()))*pt
 chosen=int(predicted>0);return dict(m=m,pilot_n=len(pilot),select_alt=chosen,predicted_gain=float(predicted),actual_increment=int(e[q1].sum()-e[q0].sum()),review=q1 if chosen else q0,stop=q0,alt=q1)

def main():
 jobs=[(n,eq.new.ROOT/('heldout_pico' if n=='heldout_pico' else 'new_crowd/'+n),False) for n in eq.new.CATALOG+('heldout_pico',)]
 jobs += [(n,eq.hist.ROOT/n,False) for n in eq.hist.TARGETS]
 jobs += [(n,extra.ROOT/loc,ext) for n,loc,ext in extra.JOBS]
 assert len(jobs)==33 and len({x[0] for x in jobs})==33
 rows=[];summary=[]
 for job,(name,path,external) in enumerate(jobs):
  with np.load(path/('scores.npz' if external else 'target.npz'),allow_pickle=False) as z:
   ranks=z['target_rank' if external else 'rank'].astype(int);e=z['target_error' if external else 'errors'].astype(bool);names=z['names'].astype(str).tolist();_,group=np.unique(z['group'] if 'group' in z else z['ids'],return_inverse=True)
  with (path/'paired.csv').open(newline='',encoding='utf8') as f:panels=[p for p in csv.DictReader(f) if float(p['budget'])==.1 and (external or p['menu']=='expanded')]
  assert len(panels)==10 and len({p['seed'] for p in panels})==10
  B=round(.1*len(e));k=int(.05*B);start=len(rows)
  for pi,p in enumerate(panels):
   ia=names.index(p['random']) if external else int(p['random']);ib=names.index(p['difficulty']) if external else int(p['difficulty'])
   a=ranks[ia][:B];b=ranks[ib][:B];mix=eq.interleave(a,b);fk=eq.fixed(a,b,e);cap,fail=eq.loss.run(a,b,e)
   dh=int(e[a].sum());mh=int(e[mix].sum());fh=int(e[fk].sum());ch=int(e[cap].sum())
   assert dh-fh<=k and dh-ch<=fail<=k
   for rep in range(30):
    seed=78562000+100000*job+1000*pi+rep;r=run(a,b,e,seed);q=r['review'];q0=r['stop'];q1=r['alt']
    assert len(q)==len(np.unique(q))==B
    if rep==0:
     altered=e.copy();unseen=np.ones(len(e),bool);unseen[q]=False;altered[unseen]=~altered[unseen];r2=run(a,b,altered,seed)
     assert np.array_equal(r2['review'],q) and r2['select_alt']==r['select_alt'] and abs(r2['predicted_gain']-r['predicted_gain'])<1e-10
    rows.append(dict(dataset=name,seed=p['seed'],rep=rep,N=len(e),B=B,k=k,default_index=ia,alternative_index=ib,same_queue=int(ia==ib),m=r['m'],pilot_n=r['pilot_n'],select_alt=r['select_alt'],predicted_gain=r['predicted_gain'],actual_increment=r['actual_increment'],default=dh,interleave=mh,fixed_k=fh,feedback=ch,paired_stop=int(e[q0].sum()),paired=int(e[q].sum()),paired_oracle=max(int(e[q0].sum()),int(e[q1].sum()))))
  subset=rows[start:];out=dict(dataset=name,N=len(e),B=B,k=k,groups=len(np.unique(group)),panels=10,reps=30,same_queue=sum(x['same_queue'] for x in subset)//30)
  for key in ('m','pilot_n','select_alt','predicted_gain','actual_increment','default','interleave','fixed_k','feedback','paired_stop','paired','paired_oracle'):
   out[key]=float(np.mean([x[key] for x in subset]))
  for key in ('default','interleave','fixed_k','feedback','paired_stop','paired_oracle'):
   out['paired_vs_'+key]=float(np.mean([x['paired']-x[key] for x in subset]))
  out['predicted_mae']=float(np.mean([abs(x['predicted_gain']-x['actual_increment']) for x in subset]));summary.append(out)
  print(name,'B',B,'D/M/K/F/P',*[round(out[x],2) for x in ('default','interleave','fixed_k','feedback','paired')],flush=True)
 for file,data in [('rows.csv',rows),('summary.csv',summary)]:
  with (HERE/file).open('w',newline='',encoding='utf8') as f:
   w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
if __name__=='__main__':main()
