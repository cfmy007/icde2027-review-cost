from pathlib import Path
import csv,numpy as np
ROOT=Path(__file__).resolve().parent

def f1(tp,fp,fn):
 den=2*tp+fp+fn
 return np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0)*100

def main():
 rows=[]
 for task in ('germs','alia'):
  with np.load(ROOT/'text_curves'/f'{task}_paired_budget_curves_v1_predictions.npz',allow_pickle=False) as z:
   pred=z['pred'];y=z['y'];names=z['arms'].tolist();_,groups=np.unique(z['groups'],return_inverse=True)
  ids=[names.index('random_0.2'),names.index('difficulty_0.2')];p=pred[:,ids,:];ng=int(groups.max()+1);counts=np.zeros((30,2,ng,3),int)
  for i in range(30):
   for j in range(2):
    for k,mask in enumerate(((p[i,j]==1)&(y==1),(p[i,j]==1)&(y==0),(p[i,j]==0)&(y==1))):counts[i,j,:,k]=np.bincount(groups,weights=mask,minlength=ng).astype(int)
  c=counts.sum(2);point=f1(c[:,:,0],c[:,:,1],c[:,:,2]).mean(0);rng=np.random.default_rng(31113131);samples=[]
  for _ in range(2000):
   src=rng.integers(10,size=10);fit=rng.integers(3,size=(10,3));runs=(src[:,None]*3+fit).ravel();w=np.bincount(rng.integers(ng,size=ng),minlength=ng);c=np.einsum('ragc,g->rac',counts[runs],w);v=f1(c[:,:,0],c[:,:,1],c[:,:,2]).mean(0);samples.append(v[1]-v[0])
  lo,hi=np.quantile(samples,[.025,.975]);rows.append(dict(dataset=task,random_f1=point[0],difficulty_f1=point[1],difference=point[1]-point[0],low=lo,high=hi))
  print(task,'random',round(point[0],2),'difficulty',round(point[1],2),'difference',round(point[1]-point[0],2),'CI',round(lo,2),round(hi,2))
 assert abs(rows[0]['difference']-5.58)<.01 and abs(rows[1]['difference']+3.61)<.01
 out=ROOT.parent/'outputs';out.mkdir(exist_ok=True)
 with (out/'text_downstream.csv').open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
