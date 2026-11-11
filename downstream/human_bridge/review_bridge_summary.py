"""Paired downstream contrasts on fixed review interventions and test items."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P = json.loads((HERE/"review_bridge_protocol.json").read_text())


def bootstrap(values, group):
    """Resample N splits and training seeds within split, then held-out images."""
    values = np.asarray(values,dtype=np.float64)
    rng = np.random.default_rng(P["bootstrap_seed"])
    groups = np.unique(group)
    ids = np.stack([np.flatnonzero(group==g) for g in groups])
    n_groups,n_within = ids.shape
    inner_weights = np.full(n_within,1/n_within)
    sims, fixed_test = [], []
    draws = P["bootstrap_draws"]
    n_images = values.shape[1]
    ordered = values[ids.ravel()]
    for start in range(0,draws,200):
        b = min(200,draws-start)
        outer = rng.integers(n_groups,size=(b,n_groups))
        inner = rng.multinomial(n_within,inner_weights,size=(b,n_groups))
        counts = np.zeros((b,n_groups,n_within),dtype=int)
        np.add.at(counts,(np.arange(b)[:,None],outer),inner)
        weights = counts.reshape(b,-1)/(n_groups*n_within)
        per_image = weights @ ordered
        fixed_test.extend(per_image.mean(1))
        sample = rng.integers(n_images,size=(b,n_images))
        sims.extend(np.take_along_axis(per_image,sample,axis=1).mean(1))
    return dict(mean=float(values.mean()),ci95=np.quantile(sims,[.025,.975]).tolist(),
                fixed_test_ci95=np.quantile(fixed_test,[.025,.975]).tolist())


def load(runs, dataset):
    arms = P["arms"]
    start = P["seed_start"]
    count = P["seed_count_per_cohort"]
    rows, packed = [], []
    for s in range(start,start+count):
        with np.load(runs/dataset/f"{s}.npz") as f:
            assert int(f["seed"])==s
            fix = list(map(int,f["fixes"]))
            high,low,checks = [int(f[k]) for k in ("high","low","budget")]
            assert fix[high]>=fix[low] and fix[4]==fix[low] and fix[0]==0
            assert checks == round(P["review_fraction"]*int(f["n"]))
            row = dict(dataset=dataset,seed=s,group=int(f["group"]),variant=str(f["variant"]),
                       n=int(f["n"]),checks=checks,common=int(f["common"]),high=high,low=low,
                       **{f"{a}_fixes":v for a,v in zip(arms,fix)})
            bits = np.unpackbits(f["hits"],axis=1,count=int(f["test_n"]))
            row.update({f"{a}_external_correct":int(v[:2000].sum()) for a,v in zip(arms,bits)})
            row.update({f"{a}_official_correct":int(v[2000:].sum()) if bits.shape[1]>2000 else ""
                        for a,v in zip(arms,bits)})
            rows.append(row)
            packed.append(f["hits"].copy())
    return rows,np.stack(packed)


def effects(rows,bits):
    group = np.array([r["group"] for r in rows])
    idx = np.arange(len(rows))
    high = np.array([r["high"] for r in rows])
    low = np.array([r["low"] for r in rows])
    n = np.array([r["n"] for r in rows])
    a = P["arms"]
    output = dict(
        seeds=len(rows), yield_per_10000={k:float(np.mean([10000*r[k+"_fixes"]/r["n"] for r in rows])) for k in a},
        difficulty_minus_random_yield=float(np.mean([10000*(r[a[3]+"_fixes"]-r[a[2]+"_fixes"])/r["n"] for r in rows])),
        mean_shared_fixes=float(np.mean([r["common"] for r in rows])),
        high_arm={a[k]:int((high==k).sum()) for k in (2,3)},
        average_review_cap=float(np.mean([r["checks"]/r["n"] for r in rows])),
    )
    intervals = {}
    for endpoint,x in (("CIFAR-10.1",bits[:,:,:2000]),("CIFAR-10",bits[:,:,2000:])):
        if x.shape[2]==0:
            continue
        z = x.astype(np.int16)
        contrasts = {
            "difficulty_minus_random":z[:,3]-z[:,2],
            "random_minus_uniform":z[:,2]-z[:,1],
            "difficulty_minus_uniform":z[:,3]-z[:,1],
            "random_minus_unreviewed":z[:,2]-z[:,0],
            "difficulty_minus_unreviewed":z[:,3]-z[:,0],
            "uniform_minus_unreviewed":z[:,1]-z[:,0],
            "high_minus_low":z[idx,high]-z[idx,low],
            "high_minus_count_matched":z[idx,high]-z[:,4],
            "count_matched_minus_low":z[:,4]-z[idx,low],
        }
        assert np.array_equal(contrasts["high_minus_count_matched"]+contrasts["count_matched_minus_low"],
                              contrasts["high_minus_low"])
        intervals[endpoint] = dict(test_n=x.shape[2],
            accuracy={k:float(x[:,j].mean()*100) for j,k in enumerate(a)},
            contrasts={k:bootstrap(v*100,group) for k,v in contrasts.items()})
    output["endpoints"] = intervals
    return output


def run(runs,out):
    out.mkdir(exist_ok=True,parents=True)
    merged, packs, summary = [], {}, {}
    for d,n_test in (("n",12000),("h",2000)):
        rows,packed = load(runs,d)
        merged.extend(rows)
        packs[d] = packed
        bits = np.unpackbits(packed,axis=2,count=n_test)
        summary[d] = effects(rows,bits)
    with (out/"runs.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)
    np.savez_compressed(out/"correctness.npz", **packs)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")


def replay(out):
    with (out/"runs.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("seed","group","n","checks","common","high","low",*(a+"_fixes" for a in P["arms"])):
            r[k] = int(r[k])
    with np.load(out/"correctness.npz") as archive:
        result = {d:effects([r for r in rows if r["dataset"]==d], np.unpackbits(archive[d],axis=2,count=n))
                  for d,n in (("n",12000),("h",2000))}
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--runs",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    replay(a.out) if a.runs is None else run(a.runs,a.out)
