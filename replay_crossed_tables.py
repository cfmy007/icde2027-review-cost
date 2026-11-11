from pathlib import Path
from collections import defaultdict
import csv,json,re,numpy as np
ROOT=Path(__file__).resolve().parent

def table(path):
 out={}
 for line in path.read_text(encoding='utf8').splitlines():
  line=re.sub(r'\\cellcolor\{[^}]*\}','',line)
  fields=[x.strip().replace('\\\\','').strip() for x in line.split('&')]
  if len(fields)>1:out[fields[0]]=[float(v) for v in fields[1:]]
 return out

def main():
 legacy=json.loads((ROOT/'crossed_study/legacy/protocol_rules.json').read_text(encoding='utf8'))['rows']
 want=table(ROOT/'paper_tables/legacy.tex');names=('CNN-64','PCA-50','RP-256')
 for label,expected in want.items():
  code=label.replace('-','_');means=[np.mean([r['proto_margin'] for r in legacy if r['repr']==name and r['injection']==code]) for name in names]
  got=means+[means[0]-means[1]]
  assert all(abs(a-b)<.0006 for a,b in zip(got,expected)),(label,got,expected)
 print('Table III: 20 values reproduced from 75 archived split records')
 expected=table(ROOT/'paper_tables/matrix_auc.tex');data=defaultdict(list)
 for ds in ('hwdb','c10','c100'):
  with (ROOT/f'data/table-5_{ds}_metrics.csv').open(encoding='utf8',newline='') as f:
   for r in csv.DictReader(f):
    if r['side']=='target' and abs(float(r['rate'])-.1)<1e-9 and abs(float(r['budget'])-.1)<1e-9:
     data[r['protocol'],ds,r['detector']].append(float(r['auc']))
 for proto,vals in expected.items():
  computed=[np.mean(data[proto,ds,det]) for ds in ('hwdb','c10','c100') for det in ('CNN','PCA','RP')]
  assert all(abs(a-b)<.0006 for a,b in zip(computed,vals)),(proto,computed,vals)
 print('Table V: all 99 AUC values reproduced from per-split metrics')
if __name__=='__main__':main()
