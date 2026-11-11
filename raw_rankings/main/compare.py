import csv,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
archive=root.parents[1]/'work/decision_closure_20261111/inputs/historical'
failed=False
for p in sorted((root/'generated').iterdir()):
    with np.load(p/'target.npz') as a, np.load(archive/p.name/'target.npz') as b:
        labels=np.array_equal(a['errors'],b['errors'])
        ranks=np.array_equal(a['rank'],b['rank'])
        agreement=(a['rank']==b['rank']).mean(axis=1).round(4).tolist() if a['rank'].shape==b['rank'].shape else []
        budgets=all(np.array_equal(a['errors'][a['rank'][:,:round(q*len(a['errors']))]].sum(1),b['errors'][b['rank'][:,:round(q*len(b['errors']))]].sum(1)) for q in (.05,.10,.20))
        groups=np.array_equal(np.unique(a['group'],return_inverse=True)[1][:,None]==np.unique(a['group'],return_inverse=True)[1],np.unique(b['group'],return_inverse=True)[1][:,None]==np.unique(b['group'],return_inverse=True)[1])
    with (p/'paired.csv').open() as a,(archive/p.name/'paired.csv').open() as b:
        source=list(csv.DictReader(a))==list(csv.DictReader(b))
    print(p.name,'labels',labels,'ranks',ranks,'groups',groups,'source',source,'budget hits',budgets,'position agreement',agreement,flush=True)
    failed |= not(labels and groups and source and budgets)
sys.exit(int(failed))
