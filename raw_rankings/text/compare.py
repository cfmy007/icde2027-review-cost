import csv
import sys
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
archive = root.parents[1]/'work/decision_closure_20261111/inputs/new_crowd'
failed = False
for path in sorted((root/'generated').iterdir()):
    if not (archive/path.name/'target.npz').exists():
        print(path.name, 'no archived target')
        continue
    with np.load(path/'target.npz') as a, np.load(archive/path.name/'target.npz') as b:
        same_errors = np.array_equal(a['errors'], b['errors'])
        same_ranks = np.array_equal(a['rank'], b['rank'])
        same_groups = np.array_equal(np.unique(a['group'],return_inverse=True)[1][:,None] == np.unique(a['group'],return_inverse=True)[1], np.unique(b['group'],return_inverse=True)[1][:,None] == np.unique(b['group'],return_inverse=True)[1])
        per_candidate = (a['rank'] == b['rank']).mean(axis=1).round(4).tolist() if a['rank'].shape == b['rank'].shape else []
    with (path/'paired.csv').open() as f, (archive/path.name/'paired.csv').open() as g:
        source = list(csv.DictReader(f)) == list(csv.DictReader(g))
    print(path.name, 'labels', same_errors, 'ranks', same_ranks, 'groups', same_groups, 'source', source, 'position agreement', per_candidate)
    failed |= not (same_errors and same_ranks and same_groups and source)
sys.exit(int(failed))
