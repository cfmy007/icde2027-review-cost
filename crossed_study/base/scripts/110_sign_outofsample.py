"""Evaluate the (p, f) sign rule on held-out data.

On a single selected set, f = corr/TP and net = (corr - dam)/n satisfy
sign(net) = sign(p - 1/(1+f)) by construction when g = 1. A match on that set
therefore does not establish predictive performance.

Estimate p and f on a labelled audit half and evaluate the predicted sign
against net change on the held-out half. The experiment uses cached probe
scores for six human-label settings, 200 random splits per setting, and the
10% budget reported in the paper. It also records the held-out damage rate g
and the identity residual sign(net) vs sign(p - g/(f + g)). Proposition 1
assumes g = 1.

Output: results/sign_prediction/result.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SETTINGS = {
    "CIFAR-100N 40.2%": "results/cifar100n/probe_CNN64.npz",
    "CIFAR-10N worse 40.2%": "results/cifar10n/probe_worse_label_CNN64.npz",
    "CIFAR-10N aggre 9.0%": "results/cifar10n/probe_aggre_label_CNN64.npz",
    "CIFAR-10N random1 17.2%": "results/noise_ladder/probe_random_label1.npz",
    "CIFAR-10N random2 18.1%": "results/noise_ladder/probe_random_label2.npz",
    "CIFAR-10N random3 17.6%": "results/noise_ladder/probe_random_label3.npz",
}

BUDGET = 0.10
N_SPLITS = 200


def selected_stats(idx, err, score, suggest, yclean, budget=BUDGET):
    k = max(1, int(round(budget * len(idx))))
    order = idx[np.argsort(-score[idx])][:k]
    e = err[order].astype(bool)
    tp = int(e.sum())
    fp = int((~e).sum())
    corr = int((e & (suggest[order] == yclean[order])).sum())
    dam = int((~e & (suggest[order] != yclean[order])).sum())
    return {"k": k, "tp": tp, "fp": fp, "corr": corr, "dam": dam,
            "precision": tp / k, "f": corr / tp if tp else 0.0,
            "g": dam / fp if fp else 0.0,
            "net_pp": 100.0 * (corr - dam) / k}


def main():
    rng = np.random.default_rng(0)
    out = {}
    for name, rel in SETTINGS.items():
        path = ROOT / rel
        if not path.exists():
            print(f"skip {name}")
            continue
        z = np.load(path)
        err = z["err"].astype(int)
        margin = z["margin"]
        suggest = z["suggest"].astype(int)
        yclean = z["yclean"].astype(int)
        n = len(err)

        hit = 0
        hit_all = 0
        ties = 0
        rows = []
        for _ in range(N_SPLITS):
            perm = rng.permutation(n)
            cal, te = perm[: n // 2], perm[n // 2:]
            a = selected_stats(cal, err, margin, suggest, yclean)
            b = selected_stats(te, err, margin, suggest, yclean)
            thr = 1.0 / (1.0 + a["f"])
            pred = np.sign(a["precision"] - thr)
            actual = np.sign(b["net_pp"])
            if pred == 0:
                pred = 1.0
            hit += int(pred == actual)
            hit_all += int(pred == actual)
            ties += int(actual == 0)
            rows.append({"pred": float(pred), "actual": float(actual),
                         "p_cal": a["precision"], "f_cal": a["f"],
                         "thr_cal": thr, "net_test": b["net_pp"],
                         "g_test": b["g"], "p_test": b["precision"]})
        prec = np.array([r["p_cal"] for r in rows])
        fcal = np.array([r["f_cal"] for r in rows])
        net = np.array([r["net_test"] for r in rows])
        gtest = np.array([r["g_test"] for r in rows])
        gt = np.array([r["g_test"] > 0 for r in rows])
        pred_pos = (prec > 1.0 / (1.0 + fcal)).mean()
        margins = np.abs(prec - 1.0 / (1.0 + fcal))
        out[name] = {
            "n": int(n), "splits": N_SPLITS, "budget": BUDGET,
            "accuracy": hit_all / N_SPLITS,
            "min_abs_margin": float(margins.min()),
            "mean_p_cal": float(prec.mean()), "mean_f_cal": float(fcal.mean()),
            "mean_threshold": float((1.0 / (1.0 + fcal)).mean()),
            "mean_net_test": float(net.mean()),
            "predicted_positive_rate": float(pred_pos),
            "actual_positive_rate": float((net > 0).mean()),
            "mean_g_test": float(gtest.mean()),
            "share_g_positive": float(gt.mean()),
        }
        print(f"{name:26s} acc {out[name]['accuracy']:.3f}  "
              f"pred+ {pred_pos:.2f} act+ {(net > 0).mean():.2f}  "
              f"p {prec.mean():.3f} f {fcal.mean():.3f} thr {(1/(1+fcal)).mean():.3f}  "
              f"net {net.mean():+.2f}pp  g {gtest.mean():.3f}")

    dst = ROOT / "results/sign_prediction"
    dst.mkdir(parents=True, exist_ok=True)
    accs = [v["accuracy"] for v in out.values()]
    out["_summary"] = {"mean_accuracy": float(np.mean(accs)),
                       "min_accuracy": float(np.min(accs)),
                       "max_accuracy": float(np.max(accs)),
                       "n_settings": len(accs)}
    (dst / "result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nmean accuracy {np.mean(accs):.3f} (min {np.min(accs):.3f}, max {np.max(accs):.3f})")
    print("wrote", dst / "result.json")


if __name__ == "__main__":
    main()
