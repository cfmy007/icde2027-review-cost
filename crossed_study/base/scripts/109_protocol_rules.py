"""Compare five label-generation rules under a matched evaluation protocol.

The circular rule differs from the minimal-pair and pixel-neighbour rules in
both its use of the evaluated representation and its nearest-class selection.
The font and source-PCA rules add nearest-class controls defined outside the
evaluated representation. All rules use the same 100 curated classes, samples,
reference/probe split, trained CNN, and 10% selection stream.

  fixed       Hand-curated minimal-pair partner; not a nearest-class rule.
  neighbor    A class among the 30 pixel neighbours; not a nearest-class rule.
  font        Nearest class by printed-glyph prototypes.
  source_pca  Nearest class computed once in PCA-50 and reused across representations.
  circular    Nearest class in the evaluated representation.

For the circular rule, the replay records conditional margin rates and
positive-margin AUC; their tie-aware decomposition is an identity on the same
scores.

Output: results/protocol_rules/result.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("cb", ROOT / "scripts" / "29_curated_bench.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)

from hcc.index import load_cache  # noqa: E402

FONTS = [r"C:\Windows\Fonts\simsun.ttc", r"C:\Windows\Fonts\simhei.ttf",
         r"C:\Windows\Fonts\simkai.ttf", r"C:\Windows\Fonts\simfang.ttf"]


def l2n(Z):
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-8)


def centroids(Z, y, n_cls):
    C = np.stack([Z[y == c].mean(0) if (y == c).any() else np.zeros(Z.shape[1])
                  for c in range(n_cls)])
    return l2n(C)


def nearest_other_from_labels(pair_of_class, y):
    return pair_of_class[y]


def nearest_other(C, Z, y):
    """Nearest class centroid other than the claimed label."""
    D = 1.0 - l2n(Z) @ C.T
    D[np.arange(len(Z)), y] = np.inf
    return D.argmin(1)


def font_pair_map(names, size=32):
    """Nearest other class by printed-glyph prototypes (representation-free)."""
    from PIL import Image, ImageDraw, ImageFont
    feats = []
    for font_path in FONTS:
        font = ImageFont.truetype(font_path, size=28)
        rows = []
        for ch in names:
            img = Image.new("L", (size, size), 0)
            d = ImageDraw.Draw(img)
            box = d.textbbox((0, 0), ch, font=font)
            w, h = box[2] - box[0], box[3] - box[1]
            d.text(((size - w) / 2 - box[0], (size - h) / 2 - box[1]),
                   ch, fill=255, font=font)
            rows.append(np.asarray(img, dtype=np.float32).ravel() / 255.0)
        feats.append(np.stack(rows))
    F = l2n(np.mean(feats, axis=0))
    D = 1.0 - F @ F.T
    np.fill_diagonal(D, np.inf)
    return D.argmin(1), F


def score_all(T_ref, ty_ref, T_probe, yn, n_cls):
    """Same four signals as 29/83, without the unused KMeans detector."""
    Cr = centroids(T_ref, ty_ref, n_cls)
    Zn = l2n(T_probe)
    D = 1.0 - Zn @ Cr.T
    proto_margin = D[np.arange(len(Zn)), yn] - D.min(1)
    nn = NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(T_ref)
    _, idx = nn.kneighbors(T_probe)
    knn_consist = 1.0 - np.mean(ty_ref[idx] == yn[:, None], axis=1)
    density = np.linalg.norm(T_probe[:, None, :] - T_ref[idx], axis=2).mean(axis=1)
    return {"proto_margin": proto_margin, "knn_consist": knn_consist, "density": density}


def q_stat(pos, neg):
    """P(pos > neg | both > 0), strict; ties counted as one half."""
    pos = pos[pos > 0]
    neg = neg[neg > 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # rank-sum form of the Mann-Whitney statistic
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def main():
    test = load_cache(ROOT / "data/processed/hwdb11_test_32.npz")
    wanted = [ch for p in cb.PAIRS for ch in p]
    sub = cb.build_named_subset(test, wanted, per_class=40, seed=0)
    names = [str(x) for x in sub["names"]]
    n_cls = len(names)
    cid = {ch: i for i, ch in enumerate(names)}
    pair_map = np.arange(n_cls)
    for p in cb.PAIRS:
        a, b = cid[p[0]], cid[p[1]]
        pair_map[a], pair_map[b] = b, a

    frozen = np.load(ROOT.parent / "inputs/font_map.npz")
    assert np.array_equal(frozen["names"], names)
    font_pair = frozen["pair_map"]
    ck = torch.load(ROOT / "results/curated_bench/model.pt", weights_only=False)
    model = cb.CNN(64, n_cls)
    model.load_state_dict(ck["state"])
    model.eval()
    X = sub["X"].astype(np.float32) / 255.0
    y = sub["y"]
    Zcnn = cb.embed_np(model, X)

    Xpix = X.reshape(len(X), -1)
    P = l2n(PCA(n_components=50, random_state=0).fit_transform(Xpix))
    rng = np.random.default_rng(0)
    R = l2n(Xpix @ rng.normal(size=(Xpix.shape[1], 256)).astype(np.float32))
    reps = {"CNN-64": l2n(Zcnn), "PCA-50": P, "RP-256": R}

    rows = []
    for seed in range(5):
        ref, probe = cb.split_ref_probe(y, n_cls, seed)
        rng2 = np.random.default_rng(seed * 100 + 100)
        sel = rng2.choice(len(probe), int(round(0.10 * len(probe))), replace=False)
        flag = np.zeros(len(probe), bool)
        flag[sel] = True
        yp = y[probe]

        nn = NearestNeighbors(n_neighbors=30, n_jobs=-1).fit(Xpix[ref])
        _, nbr = nn.kneighbors(Xpix[probe])
        neigh = np.empty(len(probe), int)
        for i in range(len(probe)):
            cand = [int(y[ref[j]]) for j in nbr[i] if int(y[ref[j]]) != int(yp[i])]
            neigh[i] = int(rng2.choice(cand)) if cand else int(yp[i])

        # frozen source: nearest other class computed once in PCA-50
        Cpca = centroids(P[ref], y[ref], n_cls)
        src_pca = nearest_other(Cpca, P[probe], yp)

        for rname, T in reps.items():
            C = centroids(T[ref], y[ref], n_cls)
            circ = nearest_other(C, T[probe], yp)
            D = 1.0 - l2n(T[probe]) @ C.T
            true_nearest = float((D.argmin(1) == yp).mean())
            for mode, src in (("fixed", pair_map[yp]), ("neighbor", neigh),
                              ("font", font_pair[yp]), ("source_pca", src_pca),
                              ("circular", circ)):
                yn = yp.copy()
                yn[sel] = src[sel]
                sc = score_all(T[ref], y[ref], T[probe], yn, n_cls)
                rec = {"seed": seed, "repr": rname, "injection": mode,
                       "true_nearest_rate": true_nearest,
                       "realized_flips": int((yn != yp).sum()), "intended_flips": len(sel)}
                for det, v in sc.items():
                    rec[det] = float(roc_auc_score(flag, v))
                if mode == "circular":
                    m = sc["proto_margin"]
                    rec["q"] = q_stat(m[flag], m[~flag])
                    a = float((m[flag] > 0).mean())
                    b = float((m[~flag] == 0).mean())
                    rec["pos_margin_rate"] = a
                    rec["neg_zero_rate"] = b
                    rec["auc_identity"] = b*(1+a)/2 + a*(1-b)*(rec["q"] or 0.)
                    assert abs(rec["auc_identity"]-rec["proto_margin"]) < 1e-10
                rows.append(rec)
        print(f"seed {seed} done", flush=True)

    out = ROOT / "results/protocol_rules"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(
        json.dumps({"rows": rows,
                    "font_pair": font_pair.tolist(),
                    "font_order": names}, indent=1), encoding="utf-8")

    print(f"\n{'repr':8s} {'rule':11s} {'margin':>7s} {'knn':>7s} {'dens':>7s} {'tau':>7s}")
    for rname in reps:
        for mode in ("fixed", "neighbor", "font", "source_pca", "circular"):
            sub_rows = [r for r in rows if r["repr"] == rname and r["injection"] == mode]
            vals = {d: np.mean([r[d] for r in sub_rows]) for d in ("proto_margin", "knn_consist", "density")}
            tn = np.mean([r["true_nearest_rate"] for r in sub_rows])
            print(f"{rname:8s} {mode:11s} {vals['proto_margin']:7.3f} {vals['knn_consist']:7.3f} "
                  f"{vals['density']:7.3f} {tn:7.3f}")
    circular = [r for r in rows if r["injection"] == "circular"]
    print("largest empirical AUC identity residual:",
          max(abs(r["auc_identity"]-r["proto_margin"]) for r in circular))
    print("minimum realized/intended flips by rule:",
          {rule: min(r["realized_flips"]/r["intended_flips"] for r in rows if r["injection"] == rule)
           for rule in sorted({r["injection"] for r in rows})})
    print("wrote", out / "result.json")


if __name__ == "__main__":
    main()

