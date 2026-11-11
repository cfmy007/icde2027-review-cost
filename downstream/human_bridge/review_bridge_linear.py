"""Second downstream learner on the same fixed review interventions."""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torchvision.models import resnet18

from review_bridge import P, episodes, get_test, interventions
from vision import WEIGHTS, embedding


def run(a):
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    xt, yt = get_test(a.dataset,a.stage,a.hroot,a.external,raw=True)
    m = resnet18(weights=None)
    m.load_state_dict(torch.load(WEIGHTS,map_location="cpu",weights_only=True))
    m.fc = torch.nn.Identity()
    m = m.to(device="cuda",memory_format=torch.channels_last).eval()
    xt = embedding(m,xt)
    out = a.out / a.dataset
    out.mkdir(exist_ok=True,parents=True)
    z = None
    cache = {}
    for j,group,v,images,ids,obs,gold,ranks in episodes(a.dataset,a.stage,a.hroot,a.nroot):
        if z is None:
            z = embedding(m,images)
        seed = P["seed_start"]+j
        path = out / f"{seed}.npz"
        if path.exists():
            continue
        labels, fixes, _, _, _, budget = interventions(obs,gold,ranks,seed)
        all_hits = []
        for arm in (0,2,3):
            key = (group,v,arm)
            if key not in cache:
                model = LogisticRegression(C=1,solver="lbfgs",max_iter=500).fit(z[ids],labels[arm])
                cache[key] = model.predict(xt)==yt
            all_hits.append(cache[key])
        np.savez_compressed(path,seed=seed,group=group,hits=np.packbits(all_hits,axis=1),test_n=len(yt),
                            checks=budget,n=len(ids),fixes=[fixes[k] for k in (0,2,3)])
        if j%10==0:
            print(a.dataset,j+1,"external hits",[int(v[:2000].sum()) for v in all_hits],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("stage","hroot","nroot","external","out"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("dataset",choices=("n","h"))
    run(p.parse_args())
