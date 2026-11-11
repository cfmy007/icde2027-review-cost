"""Benchmark noise generated from manually curated confusable character pairs.

The nearest-other-class generator previously used the evaluated representation
to create labels and then assessed them using the same prototypes. This makes
proto_margin zero for injected samples and explains the PCA AUC of 0.353. The
present benchmark uses manually selected confusable pairs independent of the
representation and compares CNN, PCA, and random projections on shared labels.

The benchmark contains 100 classes in 50 curated pairs, with 40 test samples
per class from 60 unseen writers. It compares uniformly random wrong classes,
curated confusable partners, and classes among 30 pixel-space neighbours.
The detectors are proto_margin, knn_consist, density, and cluster_dis; noise
rates are 1%, 5%, 10%, and 20%, with five split seeds. A separate audit measures
how often CosFace errors on unseen writers match the curated partners.

Outputs: results/curated_bench/result.json, embed_d64.npz, and model.pt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcc.index import load_cache  # noqa: E402
from hcc import cluster as hcc_cluster  # noqa: E402

PAIRS = ("日曰 未末 己已 人入 土士 大太 干千 王玉 木术 戌戍 折拆 暴曝 候侯 具俱 "
         "田由 甲申 白自 力刀 几九 风凤 鸟乌 兔免 狼狠 拔拨 即既 冠寇 幕慕 燥躁 "
         "籍藉 缉辑 焰馅 概慨 谦歉 微徽 撤撒 住往 佳挂 待侍 持特 泰秦 祟崇 票粟 "
         "梁粱 栽裁 戳戮 慷糠 悔诲 班斑 辨辩 疆僵").split()


def l2norm(Z):
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-8)


def build_named_subset(data, wanted, per_class=40, seed=0):
    y_all, writer_all, names = data["y"], data["writer"], data["names"]
    name_to_id = {str(n): i for i, n in enumerate(names)}
    rng = np.random.default_rng(seed)
    idx_out, kept = [], []
    for ch in wanted:
        cid = name_to_id[ch]
        idx = np.flatnonzero(y_all == cid)
        by_writer = {}
        for i in idx:
            by_writer.setdefault(int(writer_all[i]), []).append(int(i))
        writers = list(by_writer)
        rng.shuffle(writers)
        for w in writers:
            rng.shuffle(by_writer[w])
        picked, cursor = [], {w: 0 for w in writers}
        while len(picked) < per_class:
            prog = False
            for w in writers:
                if len(picked) >= per_class:
                    break
                j = cursor[w]
                if j < len(by_writer[w]):
                    picked.append(by_writer[w][j]); cursor[w] = j + 1; prog = True
            if not prog:
                break
        idx_out.extend(picked)
        kept.append(cid)
    idx_out = np.array(idx_out)
    remap = {cid: k for k, cid in enumerate(kept)}
    return {"X": data["X"][idx_out],
            "y": np.array([remap[int(y_all[i])] for i in idx_out], dtype=np.int32),
            "writer": data["writer"][idx_out],
            "names": [str(names[k]) for k in kept],
            "class_ids": np.array(kept)}


def load_train(train_caches, names, val_writers=40, seed=123):
    Xs, ys, ws = [], [], []
    offset = 0
    for p in train_caches.split(","):
        d = load_cache(Path(p))
        mp = {str(n): i for i, n in enumerate(d["names"])}
        for c, name in enumerate(names):
            if name in mp:
                m = d["y"] == mp[name]
                Xs.append(d["X"][m].astype(np.float32) / 255.0)
                ys.append(np.full(int(m.sum()), c, dtype=np.int64))
                ws.append(d["writer"][m].astype(np.int64) + offset)
        offset += int(d["writer"].max()) + 1
    X, y, w = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
    rng = np.random.default_rng(seed)
    val_w = rng.choice(np.unique(w), size=min(val_writers, len(np.unique(w)) // 5), replace=False)
    is_val = np.isin(w, val_w)
    return X[~is_val], y[~is_val], X[is_val], y[is_val]


class CNN(nn.Module):
    def __init__(self, d_z=64, n_cls=100, s=30.0, m=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, d_z)
        self.Wc = nn.Parameter(torch.randn(n_cls, d_z) * 0.1)
        self.s, self.m = s, m

    def embed(self, x):
        h = self.features(x).flatten(1)
        return F.normalize(self.fc(h), dim=1)

    def forward(self, x, y):
        z = self.embed(x)
        cos = z @ F.normalize(self.Wc, dim=1).T
        oh = F.one_hot(y, cos.size(1)).to(cos.dtype)
        return self.s * (cos - self.m * oh)


def to_tensor(X):
    return torch.from_numpy(X).unsqueeze(1)


@torch.no_grad()
def embed_np(model, X, batch=512):
    model.eval()
    T = to_tensor(X)
    return np.concatenate([model.embed(T[i:i + batch]).numpy() for i in range(0, len(T), batch)])


def kmeans_acc(Z, y, k, seeds=(0, 1)):
    from sklearn.cluster import KMeans
    return float(np.mean([hcc_cluster.evaluate(
        y, KMeans(k, n_init=5, random_state=s).fit_predict(Z))["ACC"] for s in seeds]))


def train(Xtr, ytr, Xval, yval, n_cls, args):
    torch.manual_seed(0)
    model = CNN(args.d_z, n_cls, s=30.0, m=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Ttr, Ytr = to_tensor(Xtr), torch.from_numpy(ytr)
    n = len(Ttr)
    best = {"val_kmeans": -1, "epoch": 0, "state": None, "val_acc": 0.0}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            logits = model(Ttr[idx], Ytr[idx])
            loss = F.cross_entropy(logits, Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        if ep % 2 == 0:
            Zv = embed_np(model, Xval)
            Wn = F.normalize(model.Wc, dim=1).detach().numpy()
            va = float(((Zv @ Wn.T).argmax(1) == yval).mean())
            vk = kmeans_acc(Zv, yval, n_cls)
            if vk > best["val_kmeans"]:
                best = {"val_kmeans": vk, "epoch": ep, "val_acc": va,
                        "state": {k: v.clone() for k, v in model.state_dict().items()}}
            print(f"    ep{ep:>3d} loss={tot/n:.4f} val_acc={va:.4f} val_kmeans={vk:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict(best["state"])
    return model, best


def centroids(Z, y, n_cls):
    C = np.stack([Z[y == c].mean(0) if (y == c).any() else np.zeros(Z.shape[1])
                  for c in range(n_cls)])
    return l2norm(C)


def detectors(T_ref, ty_ref, T_probe, yn, n_cls):
    Cr = centroids(T_ref, ty_ref, n_cls)
    Zn = l2norm(T_probe)
    D = 1.0 - Zn @ Cr.T
    proto_margin = D[np.arange(len(Zn)), yn] - D.min(1)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(T_ref)
    _, idx = nn.kneighbors(T_probe)
    knn_consist = 1.0 - np.mean(ty_ref[idx] == yn[:, None], axis=1)
    density = np.linalg.norm(T_probe[:, None, :] - T_ref[idx], axis=2).mean(axis=1)
    from sklearn.cluster import KMeans
    lab = KMeans(n_cls, n_init=5, random_state=0).fit_predict(T_probe)
    cluster_dis = np.zeros(len(T_probe))
    for c in np.unique(lab):
        m = lab == c
        vals, counts = np.unique(yn[m], return_counts=True)
        major = vals[counts.argmax()]
        cluster_dis[m] = 1.0 - float((yn[m] == major).mean())
    return {"proto_margin": proto_margin, "knn_consist": knn_consist,
            "density": density, "cluster_dis": cluster_dis}


def auc(score, flag):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(flag, score))


def prec_at(score, flag, frac=0.1):
    n = max(1, int(round(frac * len(score))))
    order = np.argsort(-score)[:n]
    return float(flag[order].mean())


def split_ref_probe(y, n_cls, seed):
    rng = np.random.default_rng(seed)
    ref, probe = [], []
    for c in range(n_cls):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        ref.append(idx[: len(idx) // 2])
        probe.append(idx[len(idx) // 2:])
    return np.concatenate(ref), np.concatenate(probe)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--d_z", type=int, default=64)
    ap.add_argument("--out", default="results/curated_bench")
    ap.add_argument("--rates", default="0.01,0.05,0.1,0.2")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--train", action="store_true", help="Train again instead of loading an existing model")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    test = load_cache(ROOT / "data/processed/hwdb11_test_32.npz")
    wanted = [ch for p in PAIRS for ch in p]
    sub = build_named_subset(test, wanted, per_class=40, seed=0)
    names = sub["names"]
    n_cls = len(names)
    assert n_cls == 100 and len(set(names)) == 100
    cid = {ch: i for i, ch in enumerate(names)}
    pair_map = np.arange(n_cls)
    for p in PAIRS:
        a, b = cid[p[0]], cid[p[1]]
        pair_map[a], pair_map[b] = b, a
    Xte = sub["X"].astype(np.float32) / 255.0
    yte = sub["y"]

    if args.train or not (out / "model.pt").exists():
        Xtr, ytr, Xval, yval = load_train(
            "data/processed/hwdb11_train1_100_32.npz,data/processed/hwdb11_train2_100_32.npz",
            names)
        print(f"train {Xtr.shape} | validation {Xval.shape} | test {Xte.shape} | classes {n_cls}", flush=True)
        model, best = train(Xtr, ytr, Xval, yval, n_cls, args)
        torch.save({"state": model.state_dict(), "names": names, "pair_map": pair_map,
                    "best": {k: v for k, v in best.items() if k != "state"}},
                   out / "model.pt")
        Zte = embed_np(model, Xte)
        np.savez(out / "embed_d64.npz", Zte=Zte, yte=yte, names=np.array(names))
        print(f"best epoch {best['epoch']} val_k={best['val_kmeans']:.4f}", flush=True)
    else:
        ck = torch.load(out / "model.pt", weights_only=False)
        model = CNN(args.d_z, n_cls); model.load_state_dict(ck["state"])
        Zte = np.load(out / "embed_d64.npz")["Zte"]

    Wc = F.normalize(model.Wc, dim=1).detach().numpy()
    scores = Zte @ Wc.T
    pred = scores.argmax(1)
    acc = float((pred == yte).mean())
    err = pred != yte
    partner_rate = float((pred[err] == pair_map[yte[err]]).mean()) if err.any() else 0.0
    base_rate = 1.0 / (n_cls - 1)
    print(f"test acc {acc:.4f}｜errors {int(err.sum())}｜partner rate {partner_rate:.3f} "
          f"(chance {base_rate:.3f})", flush=True)

    # Representations
    Xpix = sub["X"].reshape(len(sub["X"]), -1).astype(np.float32) / 255.0
    from sklearn.decomposition import PCA
    P = l2norm(PCA(n_components=50, random_state=0).fit_transform(Xpix))
    rng_rp = np.random.default_rng(0)
    R = l2norm(Xpix @ rng_rp.normal(size=(Xpix.shape[1], 256)).astype(np.float32))
    reps = {"CNN-64": Zte, "PCA-50": P, "RP-256": R}

    rates = [float(x) for x in args.rates.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    rows = []
    for seed in seeds:
        ref, probe = split_ref_probe(yte, n_cls, seed)
        # Fixed pixel-neighbour source: use reference pixels and labels.
        from sklearn.neighbors import NearestNeighbors
        nn_pix = NearestNeighbors(n_neighbors=30, n_jobs=-1).fit(Xpix[ref])
        _, nbr = nn_pix.kneighbors(Xpix[probe])
        for rate in rates:
            rng = np.random.default_rng(seed * 100 + int(rate * 1000))
            sel = rng.choice(len(probe), int(round(rate * len(probe))), replace=False)
            for mode in ("random", "confusable", "neighbor"):
                yn = yte[probe].copy()
                if mode == "random":
                    for i in sel:
                        c = int(rng.integers(0, n_cls - 1))
                        yn[i] = c if c < yte[probe][i] else c + 1
                elif mode == "confusable":
                    yn[sel] = pair_map[yte[probe][sel]]
                else:
                    for k, i in enumerate(sel):
                        others = [int(yte[ref[j]]) for j in nbr[i]
                                  if int(yte[ref[j]]) != int(yte[probe][i])]
                        yn[i] = int(rng.choice(others)) if others else int(yte[probe][i])
                flag = np.zeros(len(probe), bool); flag[sel] = True
                for rname, T in reps.items():
                    sc = detectors(T[ref], yte[ref], T[probe], yn, n_cls)
                    for k, v in sc.items():
                        rows.append({"seed": seed, "rate": rate, "mode": mode, "repr": rname,
                                     "detector": k, "auc": auc(v, flag),
                                     "p10": prec_at(v, flag)})
            print(f"seed {seed} rate {rate} done", flush=True)

    (out / "result.json").write_text(json.dumps(
        {"acc": acc, "errors": int(err.sum()), "partner_rate": partner_rate,
         "chance": base_rate, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out / "result.json")


if __name__ == "__main__":
    main()
