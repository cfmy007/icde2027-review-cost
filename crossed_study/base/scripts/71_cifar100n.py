"""CIFAR-100N: real human label noise, no injection.

Protocol
  - CIFAR-100 train images (50k), human labels from CIFAR-100N (40.2% errors).
  - 80/20 split by human label: 80% reference (prototypes + CNN training),
    20% probe (detection; clean CIFAR labels only for scoring).
  - Representations: PCA-50, RP-256, CNN-64 trained on the 80% human labels.
  - Signals: prototype margin, kNN consistency, density, confidence.
  - Positives: human label != clean label; negatives: equal.
  - AUC + bootstrap CI; top-10% repair using clean labels for scoring.

Output: results/cifar100n/result.json
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data/raw/cifar100/cifar-100-python"


def l2n(Z):
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-8)


def load_cifar():
    with open(DATA / "train", "rb") as f:
        tr = pickle.load(f, encoding="bytes")
    X = tr[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y = np.array(tr[b"fine_labels"])
    return X, y


class CNN(nn.Module):
    def __init__(self, n_classes, dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, dim)
        self.Wc = nn.Parameter(torch.randn(n_classes, dim) * 0.01)

    def embed(self, x):
        return F.normalize(self.fc(self.features(x).flatten(1)), dim=1)

    def forward(self, x, y):
        z = self.embed(x)
        cos = z @ F.normalize(self.Wc, dim=1).T
        onehot = F.one_hot(y, self.Wc.shape[0]).float()
        return 30.0 * (cos - 0.5 * onehot)


def train_cnn(X, y, n_classes, epochs=8, seed=0):
    torch.manual_seed(seed)
    model = CNN(n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y.astype(np.int64))
    n = len(X)
    for ep in range(epochs):
        perm = torch.from_numpy(rng.permutation(n))
        total = 0.0
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xb, yb = Xt[idx], yt[idx]
            if rng.random() < 0.5:
                xb = torch.flip(xb, dims=[3])
            pad = F.pad(xb, (4, 4, 4, 4), mode="reflect")
            ox, oy = int(rng.integers(0, 9)), int(rng.integers(0, 9))
            xb = pad[:, :, oy:oy + 32, ox:ox + 32]
            opt.zero_grad()
            out = model(xb, yb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        print(f"  epoch {ep+1}/{epochs} loss {total/n:.4f}", flush=True)
    model.eval()
    return model


def embed(model, X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out.append(model.embed(torch.from_numpy(X[i:i + 256])).numpy())
    return np.concatenate(out)


def centroids(Z, y, n_cls):
    C = []
    for c in range(n_cls):
        pts = Z[y == c]
        C.append(pts.mean(0) if len(pts) else Z.mean(0))
    return l2n(np.stack(C))


def score_signals(Zref, yref, Zprobe, yprobe, n_cls, k=10):
    C = centroids(Zref, yref, n_cls)
    D = 1.0 - l2n(Zprobe) @ C.T
    claim = D[np.arange(len(Zprobe)), yprobe]

    nn = NearestNeighbors(n_neighbors=min(k, len(Zref))).fit(Zref)
    dist, idx = nn.kneighbors(Zprobe)
    knn = 1.0 - (yref[idx] == yprobe[:, None]).mean(1)
    vote = np.array([np.bincount(yref[idx[i]], minlength=n_cls).argmax() for i in range(len(idx))])
    density = dist.mean(1)

    return {
        "margin": claim - D.min(1),
        "knn": knn,
        "density": density,
        "vote": vote,
        "centroids": C,
        "D": D,
    }


def auc_ci(y, s, n_boot=2000, seed=0):
    auc = float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return auc, float(lo), float(hi)


def main() -> None:
    t0 = time.time()
    X, yclean = load_cifar()
    d = torch.load(ROOT / "data/raw/cifar100n/CIFAR-100_human.pt",
                   map_location="cpu", weights_only=False)
    yhuman = np.array(d["noisy_label"])
    assert len(yhuman) == len(yclean) == 50000
    err = yhuman != yclean
    print(f"human noise {err.mean():.3f} ({err.sum()}/50000)", flush=True)

    # 80/20 split by human label
    rng = np.random.default_rng(0)
    ref, probe = [], []
    for c in range(100):
        idx = np.flatnonzero(yhuman == c)
        rng.shuffle(idx)
        n_ref = int(round(0.8 * len(idx)))
        ref.extend(idx[:n_ref]); probe.extend(idx[n_ref:])
    ref = np.array(ref); probe = np.array(probe)
    print(f"ref {len(ref)} probe {len(probe)}; probe errors {err[probe].sum()}", flush=True)

    n_cls = 100
    Xr = X[ref]; Xp = X[probe]
    yr = yhuman[ref]; yp = yhuman[probe]
    ep = err[probe].astype(int)

    # PCA / RP on all train images (label-free)
    flat = X.reshape(len(X), -1)
    pca = PCA(n_components=50, svd_solver="randomized", random_state=0).fit(flat)
    Zpca = l2n(pca.transform(flat))
    rng_rp = np.random.default_rng(0)
    R = rng_rp.normal(0, 1 / np.sqrt(flat.shape[1]), size=(flat.shape[1], 256)).astype(np.float32)
    Zrp = l2n(flat @ R)
    print(f"pca/rp done {time.time()-t0:.0f}s", flush=True)

    # CNN on reference human labels
    cnn = train_cnn(Xr, yr, n_cls, epochs=8)
    Zcnn = l2n(embed(cnn, X))
    print(f"cnn done {time.time()-t0:.0f}s", flush=True)

    out = {"noise_rate": float(err.mean()), "n_ref": int(len(ref)), "n_probe": int(len(probe)),
           "n_probe_err": int(ep.sum()), "signals": {}}
    for name, Z in [("PCA-50", Zpca), ("RP-256", Zrp), ("CNN-64", Zcnn)]:
        sc = score_signals(Z[ref], yr, Z[probe], yp, n_cls)
        row = {}
        for sig in ("margin", "knn", "density"):
            auc, lo, hi = auc_ci(ep, sc[sig])
            row[sig] = {"auc": auc, "ci": [lo, hi]}
        if name == "CNN-64":
            W = F.normalize(cnn.Wc, dim=1).detach().numpy()
            logits = Z[probe] @ W.T
            prob = np.exp(logits - logits.max(1, keepdims=True))
            prob = prob / prob.sum(1, keepdims=True)
            conf = 1.0 - prob[np.arange(len(probe)), yp]
            auc, lo, hi = auc_ci(ep, conf)
            row["confidence"] = {"auc": auc, "ci": [lo, hi]}
            sc["confidence"] = conf
        # repair at top 10%, clean labels only for scoring
        n_sel = int(round(0.10 * len(probe)))
        for sig in (("margin", "confidence") if name == "CNN-64" else ("margin",)):
            sel = np.argsort(-sc[sig])[:n_sel]
            sug = sc["D"][sel].argmin(1)
            corrected = int(((sug == yclean[probe][sel]) & (ep[sel] == 1)).sum())
            damaged = int(((sug != yclean[probe][sel]) & (ep[sel] == 0)).sum())
            row[f"repair_{sig}"] = {
                "selected": n_sel, "corrected": corrected, "damaged": damaged,
                "net_pp": 100.0 * (corrected - damaged) / len(probe),
                "precision": float((ep[sel] == 1).mean()),
            }
        out["signals"][name] = row
        np.savez_compressed(
            ROOT / "results/cifar100n" / f"probe_{name.replace('-', '')}.npz",
            margin=sc["margin"], knn=sc["knn"], density=sc["density"],
            conf=sc.get("confidence", np.full(len(probe), np.nan)),
            suggest=sc["D"].argmin(1), vote=sc["vote"],
            yhuman=yp, yclean=yclean[probe], err=ep, probe=probe,
        )
        print(name, {k: (round(v["auc"], 4) if isinstance(v, dict) and "auc" in v else v)
                     for k, v in row.items()}, flush=True)

    dst = ROOT / "results/cifar100n"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{time.time()-t0:.0f}s -> {dst}/result.json")


if __name__ == "__main__":
    main()


