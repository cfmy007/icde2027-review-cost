from pathlib import Path
from collections import defaultdict
import csv,subprocess,sys,numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from review_cost import cumulative,signed_debt

def main():
 output=ROOT/'outputs/review_metrics';output.mkdir(parents=True,exist_ok=True)
 subprocess.run([sys.executable,str(ROOT/'code/review_cost.py'),str(ROOT/'work/decision_closure_20261111/inputs/historical'),str(output),'--metrics-only'],check=True)
 with (output/'paired_costs.csv').open(encoding='utf8',newline='') as f:generated=list(csv.DictReader(f))
 with (ROOT/'data/table-1_paired_costs.csv').open(encoding='utf8',newline='') as f:old={(r['dataset'],r['seed'],r['budget']):r for r in csv.DictReader(f) if r['reference']=='1'}
 fields=('checks','random_detector','difficulty_detector','random_hits','difficulty_hits','gap','signed_extra_checks','exclusive_each')
 for r in generated:
  previous=old[r['dataset'],r['seed'],r['budget']]
  for field in fields:assert r[field]==previous[field],(r['dataset'],r['seed'],field)
 print('Table I: 510 source-seed/budget records across 17 targets reproduced')
 expected={'16-two-targets__results_ag':(568.0,567.2,.8),'16-two-targets__results_pgr':(84.8,85.0,-.2)}
 for name,want in expected.items():
  path=ROOT/'work/decision_closure_20261111/inputs/heldout'/name
  with np.load(path/'scores.npz',allow_pickle=False) as z:
   ranks=z['target_rank'];errors=z['target_error'].astype(bool);names=z['names'].tolist()
  with (path/'paired.csv').open(encoding='utf8',newline='') as f:panels=[r for r in csv.DictReader(f) if float(r['budget'])==.1]
  b=round(.1*len(errors));values=[]
  for p in panels:
   a=ranks[names.index(p['random'])];c=ranks[names.index(p['difficulty'])];ca=cumulative(errors,a);cc=cumulative(errors,c)
   values.append((ca[b],cc[b],signed_debt(ca,cc,b)))
  got=np.mean(values,axis=0)
  assert np.allclose(got,want,atol=.05),(name,got,want)
  print(name,'10 source choices reproduced',*[round(float(v),1) for v in got])
if __name__=='__main__':main()
