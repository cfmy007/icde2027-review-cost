"""Recompute the second learner's paired accuracy estimates."""
import argparse
import json
from pathlib import Path

import numpy as np

from review_bridge_summary import P, bootstrap


def run(runs,out):
    out.mkdir(exist_ok=True,parents=True)
    path = out/"correctness.npz"
    if runs is not None:
        packed = {}
        for d in ("n","h"):
            for key in ("hits","group"):
                collected = []
                for seed in range(P["seed_start"],P["seed_start"]+P["seed_count_per_cohort"]):
                    with np.load(runs/d/f"{seed}.npz") as f:
                        assert int(f["seed"])==seed
                        collected.append(f[key].copy())
                packed[d+"_"+key] = np.stack(collected)
        np.savez_compressed(path,**packed)
    arms = ("unreviewed","random_mask_choice","difficulty_mask_choice")
    summary = {}
    with np.load(path) as f:
        for d,n in (("n",12000),("h",2000)):
            group = f[d+"_group"]
            b = np.unpackbits(f[d+"_hits"],axis=2,count=n)
            by_endpoint = {}
            for name,x in (("CIFAR-10.1",b[:,:,:2000]),("CIFAR-10",b[:,:,2000:])):
                if x.shape[2]:
                    z = x.astype(np.int16)
                    by_endpoint[name] = dict(accuracy={a:float(x[:,i].mean()*100) for i,a in enumerate(arms)},
                       contrasts={k:bootstrap((z[:,i]-z[:,j])*100,group) for k,i,j in
                                  (("difficulty_minus_random",2,1),("random_minus_unreviewed",1,0),
                                   ("difficulty_minus_unreviewed",2,0))})
            summary[d] = by_endpoint
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--runs",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.runs,a.out)
