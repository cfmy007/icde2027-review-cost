from pathlib import Path
import csv,json,numpy as np
ROOT=Path(__file__).resolve().parent/'work/decision_closure_20261111/inputs'

def main():
 paths=sorted((ROOT/'historical').glob('*/paired.csv'))+sorted((ROOT/'new_crowd').glob('*/paired.csv'))+[ROOT/'heldout_pico/paired.csv']+sorted((ROOT/'heldout').glob('*/paired.csv'))
 count=0
 for path in paths:
  external=(path.parent/'scores.npz').exists();dataset=path.parent.name
  with np.load(path.parent/('scores.npz' if external else 'target.npz'),allow_pickle=False) as z:names=z['names'].astype(str).tolist()
  with path.open(encoding='utf8',newline='') as f:
   for row in csv.DictReader(f):
    if not external and row['menu']!='expanded':continue
    for side in ('random','difficulty'):
     score=np.array(json.loads(row[side+'_source']));selection=names[int(np.argmax(score))] if external else str(int(np.argmax(score)))
     assert selection==row[side],(dataset,row['seed'],row['budget'],side,selection,row[side]);count+=1
 print('Source selections reproduced from',count,'source-score vectors')
if __name__=='__main__':main()
