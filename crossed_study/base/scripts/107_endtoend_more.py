"""More end-to-end seeds and a second architecture.

Phase A: 10 seeds, 3-conv downstream, 12 epochs, CIFAR-10N worse.
Phase B: 3 seeds, 4-conv downstream, 12 epochs, same setting.

Detector signals are computed once (two-fold out-of-sample, 8-epoch CNN).
Output: results/endtoend_more/result.json
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("c100n", ROOT / "scripts" / "71_cifar100n.py")
c100n = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c100n)

DATA = ROOT / "data/raw/cifar10/cifar-10-batches-py"
HUMAN = ROOT / "data/raw/cifar10n/CIFAR-10_human.pt"
BUDGETS = np.round(np.arange(0.01, 0.501, 0.005), 3)
TARGET = 0.70
EPOCHS = 12


def load_cifar10():
    Xs, ys = [], []
    for i in range(1, 6):
        with open(DATA / f"data_batch_{i}", "rb") as f:
            d = pickle.load(f, encoding="bytes")
        Xs.append(d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0)
        ys.append(np.array(d[b"labels"]))
    with open(DATA / "test_batch", "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return (np.concatenate(Xs), np.concatenate(ys),
            d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0,
            np.array(d[b"labels"]))


class PlainCNN(torch.nn.Module):
    def __init__(self, n_classes=10, deep=False):
        super().__init__()
        layers = [torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
                  torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
                  torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU()]
        if deep:
            layers += [torch.nn.Conv2d(128, 256, 3, padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU()]
        layers += [torch.nn.AdaptiveAvgPool2d(1)]
        self.features = torch.nn.Sequential(*layers)
        self.fc = torch.nn.Linear(256 if deep else 128, n_classes)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


def train_plain(X, y, seed, deep=False):
    torch.manual_seed(seed)
    model = PlainCNN(deep=deep)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.int64))
    n = len(X)
    for ep in range(EPOCHS):
        perm = torch.from_numpy(rng.permutation(n))
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xb, yb = Xt[idx], yt[idx]
            if rng.random() < 0.5:
                xb = torch.flip(xb, dims=[3])
            pad = F.pad(xb, (4, 4, 4, 4), mode="reflect")
            ox, oy = int(rng.integers(0, 9)), int(rng.integers(0, 9))
            xb = pad[:, :, oy:oy + 32, ox:ox + 32]
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    model.eval()
    return model


def l2n(Z):
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-8)


def centroids(Z, y, n_cls=10):
    C = np.zeros((n_cls, Z.shape[1]), dtype=np.float32)
    for c in range(n_cls):
        m = y == c
        if m.any():
            C[c] = Z[m].mean(0)
    return l2n(C)


def signals(model, Xref, yref, Xq, yq):
    with torch.no_grad():
        Zref = l2n(model.embed(torch.from_numpy(Xref)).numpy())
        Zq = l2n(model.embed(torch.from_numpy(Xq)).numpy())
    D = 1.0 - Zq @ centroids(Zref, yref).T
    margin = D[np.arange(len(Zq)), yq] - D.min(1)
    return margin, D.argmin(1)


def pick_budget(err_audit, score_audit):
    order = np.argsort(-score_audit)
    hits = np.cumsum(err_audit[order])
    best, n = 0.0, len(err_audit)
    for b in BUDGETS:
        k = max(1, int(round(b * n)))
        if k < 20:
            continue
        if hits[k - 1] / k < TARGET:
            break
        best = b
    return best


def run_phase(X, yh, yclean, Xte, yte, margin, suggest, seeds, deep, tag, out, dst):
    err = (yh != yclean).astype(int)
    runs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        aud = rng.choice(len(X), int(round(0.10 * len(X))), replace=False)
        b = pick_budget(err[aud], margin[aud])
        k = int(round(b * len(X)))
        top = np.argsort(-margin)[:k]
        top10 = np.argsort(-margin)[: int(round(0.10 * len(X)))]
        sets = {"none": yh.copy(), "fixed10": yh.copy(), "calib": yh.copy(), "review": yh.copy()}
        sets["fixed10"][top10] = suggest[top10]
        sets["calib"][top] = suggest[top]
        sets["review"][top] = yclean[top]
        acc = {}
        for name, lab in sets.items():
            model = train_plain(X, lab, seed, deep=deep)
            with torch.no_grad():
                acc[name] = float((model(torch.from_numpy(Xte)).argmax(1).numpy() == yte).mean())
            print(f"{tag} seed {seed} {name} {acc[name]:.4f}", flush=True)
        runs.append({"seed": seed, "budget": b, "acc": acc})
        out[tag] = runs
        (dst / "result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def main():
    t0 = time.time()
    X, yclean, Xte, yte = load_cifar10()
    d = torch.load(HUMAN, map_location="cpu", weights_only=False)
    yh = np.array(d["worse_label"])
    n = len(X)
    perm = np.random.default_rng(0).permutation(n)
    folds = [perm[: n // 2], perm[n // 2:]]
    margin = np.zeros(n, dtype=np.float32)
    suggest = np.zeros(n, dtype=np.int64)
    for i, te in enumerate(folds):
        tr = folds[1 - i]
        model = c100n.train_cnn(X[tr], yh[tr], 10, epochs=8, seed=i)
        m, s = signals(model, X[tr], yh[tr], X[te], yh[te])
        margin[te], suggest[te] = m, s
        print(f"detector {i} done {time.time()-t0:.0f}s", flush=True)

    dst = ROOT / "results/endtoend_more"
    dst.mkdir(parents=True, exist_ok=True)
    out = {}
    run_phase(X, yh, yclean, Xte, yte, margin, suggest, range(10), False, "10seed_3conv", out, dst)
    run_phase(X, yh, yclean, Xte, yte, margin, suggest, range(3), True, "3seed_4conv", out, dst)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
