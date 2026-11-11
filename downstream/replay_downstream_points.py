from pathlib import Path
import csv,json,numpy as np
from summarize_equal_cost_selection import metrics
ROOT=Path(__file__).resolve().parent

def main():
 output=[]
 for dataset in ('c100','wiki'):
  for learner in ('linear','mlp'):
   folder=ROOT/'equal_cost_selection'/dataset/learner
   collected={}
   for f in sorted(folder.glob('*.npz')):
    with np.load(f,allow_pickle=False) as z:
     names=z['policies'].tolist();p=np.unpackbits(z['predictions'],axis=1,count=int(z['n_test'])).astype(bool) if dataset=='c100' else z['predictions']
     score=metrics(p[None,:,:],z['test_reference'])[0]*100;repairs=z['repairs']
     for i,name in enumerate(names):
      if not name.startswith('oracle/'):
       collected.setdefault(name,[]).append((float(score[i]),int(repairs[i])))
   expected=json.loads((ROOT.parent/'data'/f'table-8_{dataset}_{learner}.json').read_text(encoding='utf8'))['policy']
   for name,values in collected.items():
    assert len(values)==100
    quality,repairs=np.mean(values,axis=0)
    assert abs(quality-expected[name]['quality_percent'])<1e-8,(dataset,learner,name,quality,expected[name])
    assert abs(repairs-expected[name]['mean_repairs'])<1e-8
    output.append(dict(dataset=dataset,learner=learner,policy=name,runs=len(values),quality_percent=quality,mean_repairs=repairs))
   print(dataset,learner,'100 paired runs, point estimates reproduced')
 out=ROOT.parent/'outputs';out.mkdir(exist_ok=True)
 with (out/'downstream_point_estimates.csv').open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=output[0]);w.writeheader();w.writerows(output)
if __name__=='__main__':main()
