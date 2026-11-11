"""Paired downstream replay of the actual source-selected human review queues."""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STUDY = Path(os.environ.get("ICDE_STUDY", HERE.parent / "study")).resolve()
sys.path.insert(0, str(STUDY))
from downstream100 import import_script, padded_images, schedule, train_pair_arm

P = json.loads((HERE / "review_bridge_protocol.json").read_text())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def episodes(dataset, stage, hroot, nroot):
    prior = module(nroot / "audit.py", "target_audit")
    p = json.loads((nroot / "protocol.json").read_text())
    if dataset == "n":
        lookup = {(s,v): (e,r) for v,s,e,r in prior.episodes_n(nroot,p)}
        with np.load(nroot / "n_inputs.npz") as src, np.load(stage / "c10_pixels.npz") as pixels:
            rest, y, x = [pixels[k] for k in ("rest", "y", "images")]
            assert np.array_equal(y[rest], src["y"])
            seeds = list(src["seeds"])
            variants = p["target_n_variants"]
            for j in range(P["seed_count_per_cohort"]):
                split = int(seeds[j % len(seeds)])
                v = variants[(j//len(seeds) + j%len(seeds)) % len(variants)]
                ids = rest[src[f"{split}_target"]]
                obs, gold = src[f"{split}_{v}"], y[ids]
                e,r = lookup[split,v]
                assert np.array_equal(e, obs!=gold)
                yield j, split, v, x, ids, obs, gold, r
    else:
        with np.load(hroot / "images.npz") as data:
            keep = data["keep"]
            x = data["target_x"][keep].reshape(-1,3,32,32)
            gold = data["target_y"][keep]
        counts = np.load(hroot / "counts.npy")[keep].cumsum(1)
        for j,(variant,draw,e,r) in enumerate(prior.episodes_h(hroot,p)):
            rng = np.random.default_rng(draw)
            obs = (rng.integers(0,counts[:,-1])[:,None]>=counts).sum(1)
            assert np.array_equal(e, obs!=gold)
            yield j, draw, variant, x, np.arange(len(x)), obs, gold, r


def interventions(obs, gold, ranks, seed):
    budget = round(P["review_fraction"]*len(obs))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 701]))
    queues = [rng.choice(len(obs), budget, replace=False), ranks[0][:budget], ranks[1][:budget]]
    fixes = [q[obs[q]!=gold[q]] for q in queues]
    high = 1 + int(len(fixes[2])>len(fixes[1]))
    low = 3-high
    common = np.intersect1d(fixes[high],fixes[low])
    unique = np.setdiff1d(fixes[high],common)
    thin = np.r_[common, rng.choice(unique, len(fixes[low])-len(common), replace=False)]
    arm_ids = [np.array([],dtype=int), *fixes, thin]
    labels = [obs.copy() for _ in arm_ids]
    for lab, ids in zip(labels, arm_ids):
        lab[ids] = gold[ids]
        assert int((lab==gold).sum()-(obs==gold).sum()) == len(ids)
    return labels, [len(ids) for ids in arm_ids], high+1, low+1, len(common), budget


def get_test(dataset, stage, hroot, external, raw=False):
    file_x, file_y = [external / f"cifar10.1_v6_{k}.npy" for k in ("data","labels")]
    x1 = np.load(file_x).transpose(0,3,1,2).copy()
    y1 = np.load(file_y)
    with np.load(stage / "c10_pixels.npz") as train, np.load(hroot / "images.npz") as test:
        external_images = {r.tobytes() for r in x1}
        assert len(x1) == 2000 and len(external_images) == len(x1)
        assert not any(r.tobytes() in external_images for r in train["images"])
        orig = test["target_x"].reshape(-1,3,32,32)
        assert not any(r.tobytes() in external_images for r in orig)
        if dataset == "n":
            x1 = np.concatenate((x1,orig))
            y1 = np.concatenate((y1,test["target_y"]))
    if raw:
        return x1,y1
    xt = torch.from_numpy(x1).to("cuda").float().div_(255).contiguous(memory_format=torch.channels_last)
    yt = torch.as_tensor(y1,device="cuda",dtype=torch.long)
    return xt,yt


def run(args):
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original = import_script(STUDY / "base/scripts/107_endtoend_more.py")
    xt,yt = get_test(args.dataset,args.stage,args.hroot,args.external)
    out = args.out / args.dataset
    out.mkdir(exist_ok=True,parents=True)
    for j,group,v,x,ids,obs,gold,ranks in episodes(args.dataset,args.stage,args.hroot,args.nroot):
        seed = P["seed_start"]+j
        path = out / f"{seed}.npz"
        if path.exists():
            continue
        if args.limit is not None and j >= args.limit:
            break
        labels, fixes, high, low, common, budget = interventions(obs,gold,ranks,seed)
        plans = schedule(len(ids),seed,"cuda")
        padded = padded_images(x[ids],"cuda")
        outcomes = []
        for lab in labels:
            value = torch.as_tensor(lab,device="cuda",dtype=torch.long)
            outcomes.append(train_pair_arm(original,padded,value,plans,seed,xt,yt,"cuda",return_hits=True))
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, seed=seed, group=group, variant=v, n=len(ids), budget=budget,
                            fixes=fixes, high=high, low=low, common=common, hits=np.packbits(outcomes,axis=1),
                            test_n=len(yt))
        tmp.replace(path)
        print(args.dataset, j+1, seed, "fixes", fixes, "external hits", [int(a[:2000].sum()) for a in outcomes],flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("stage","hroot","nroot","external","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("dataset",choices=("n","h"))
    p.add_argument("--limit",type=int)
    run(p.parse_args())
