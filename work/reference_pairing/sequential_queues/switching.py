from pathlib import Path
from collections import defaultdict,deque
import csv,sys,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import staged_review as base
HERE=Path(__file__).resolve().parent

def run(a,b,truth,margin,trace=False):
 k=len(a);batch=max(1,int(.02*k));queues=(a,b);seen=np.zeros(len(truth),bool);pos=[0,0];hist=[deque(maxlen=100),deque(maxlen=100)];hits=n=t=0;chosen=[0,0];checked=[]
 while n<k:
  avail=[]
  for j,q in enumerate(queues):
   while pos[j]<k and seen[q[pos[j]]]:pos[j]+=1
   if pos[j]<k:avail.append(j)
  if len(avail)==1:arm=avail[0]
  elif t<2:arm=t
  else:arm=int(np.mean(hist[1])>np.mean(hist[0])+margin)
  q=queues[arm];take=[]
  while pos[arm]<k and len(take)<min(batch,k-n):
   item=int(q[pos[arm]]);pos[arm]+=1
   if not seen[item]:seen[item]=True;take.append(item)
  v=truth[take].astype(int);hist[arm].extend(v.tolist());hits+=int(v.sum());n+=len(take);chosen[arm]+=len(take);t+=1;checked.extend(take)
 return (hits,chosen,checked) if trace else (hits,chosen)

def main():
 rows=[]
 for name,panel,group,a,b,e in base.cases():
  direct=int(e[a].sum());mix=int(e[base.interleave(a,b)].sum())
  for margin in (0.,.05,.1):
   hits,chosen=run(a,b,e,margin)
   rows.append(dict(dataset=name,panel=panel,group=group,budget=len(a),margin=margin,hits=hits,default=direct,mix=mix,from_default=chosen[0],from_alternative=chosen[1]))
 with (HERE/'switching_rows.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 by=defaultdict(list)
 for r in rows:by[r['dataset'],r['margin']].append(r)
 out=[]
 for (name,margin),rs in by.items():
  ids=sorted({r['group'] for r in rs});d=dict(dataset=name,margin=margin,budget=rs[0]['budget'],groups=len(ids),panels=len(rs))
  for key in ('hits','default','mix','from_default','from_alternative'):
   d[key]=float(np.mean([np.mean([r[key] for r in rs if r['group']==g]) for g in ids]))
  out.append(d)
 with (HERE/'switching_summary.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
if __name__=='__main__':main()
