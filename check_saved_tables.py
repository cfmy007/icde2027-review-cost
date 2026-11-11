"""Recompute the paper's headline tables from the files in data/.

Run from the artifact root:  python reproduce.py
Every check prints PASS or a diff. No network, no dataset, no training.
"""
from pathlib import Path
import csv
import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE / "data"
fails = []


def chk(name, got, want, tol=0.051):
    ok = abs(got - want) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got:.2f} vs {want}")
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------- Table I
print("Table I  (review cost at a 10% budget, 17 targets)")
rows = [r for r in csv.DictReader((D / "table-1_paired_costs.csv").open(newline="", encoding="utf8"))
        if abs(float(r["budget"]) - 0.1) < 1e-9 and r["reference"] == "1"]
by = {}
for r in rows:
    by.setdefault(r["dataset"], []).append(r)
print(f"  draws: {len(rows)} (expect 170)")
if len(rows) != 170:
    fails.append("row count")

WANT = {"Unhealthy-Comments": (449, 346.0, 193.0, 294.0), "HateXplain": (319, 204.6, 156.0, 129.8),
        "D3code": (116, 78.3, 55.1, 45.6), "GoEmotions": (404, 316.9, 278.2, 104.7),
        "SBIC": (600, 283.7, 269.0, 46.9), "CIFAR-10N": (600, 542.0, 521.0, 28.7)}
for ds, (B, R, H, extra) in WANT.items():
    rs = by[ds]
    chk(f"{ds} B", max(int(float(r["checks"])) for r in rs), B, 0)
    chk(f"{ds} R hits", np.mean([float(r["random_hits"]) for r in rs]), R)
    chk(f"{ds} H hits", np.mean([float(r["difficulty_hits"]) for r in rs]), H)
    chk(f"{ds} extra checks", np.mean([float(r["signed_extra_checks"]) for r in rs]), extra)

sw = sum(1 for r in rows if r["random_detector"] != r["difficulty_detector"])
chk("detector switches", sw, 85, 0)
eq = sum(1 for r in rows if abs(float(r["gap"])) < 1e-9)
chk("equal-yield draws", eq, 96, 0)
mat = sum(1 for r in rows if float(r["signed_extra_checks"]) >= int(r["material_checks"]))
chk("material-cost draws", mat, 28, 0)

# ---------------------------------------------------------------- Table X tally
print("\nTable X  (four policies on 33 targets)")
eq_rows = []
for nm in ("table-10_equal_risk_summary.csv", "table-10_equal_risk_extra.csv"):
    eq_rows += list(csv.DictReader((D / nm).open(newline="", encoding="utf8")))
hi = sum(1 for r in eq_rows if float(r["feedback"]) - float(r["fixed"]) > 1e-9)
lo = sum(1 for r in eq_rows if float(r["feedback"]) - float(r["fixed"]) < -1e-9)
sa = len(eq_rows) - hi - lo
print(f"  targets {len(eq_rows)}; cap-limited vs fixed-k: {hi} better / {sa} equal / {lo} worse")
if (hi, sa, lo) != (6, 22, 5):
    fails.append("table X tally")
inter = sum(1 for r in eq_rows
            if float(r["interleave"]) >= max(float(r["default"]), float(r["fixed"]), float(r["feedback"])) - 1e-9)
chk("targets where interleave is tied or best", inter, 21, 0)

# ---------------------------------------------------------------- Table V AUC
print("\nTable V  (target ROC-AUC at 10% corruption, NC-CNN row)")
WANT_AUC = {"hwdb": (0.891, 0.672, 0.654), "c10": (0.689, 0.568, 0.574), "c100": (0.266, 0.426, 0.480)}
for ds, want in WANT_AUC.items():
    rs = [r for r in csv.DictReader((D / f"table-5_{ds}_metrics.csv").open(newline="", encoding="utf8"))
          if r["side"] == "target" and abs(float(r["rate"]) - 0.1) < 1e-9
          and float(r["budget"]) == 0.1 and r["protocol"] == "NC-CNN"]
    for det, w in zip(("CNN", "PCA", "RP"), want):
        vals = [float(r["auc"]) for r in rs if r["detector"] == det]
        chk(f"{ds} NC-CNN {det} AUC", float(np.mean(vals)), w, 0.0006)

# ---------------------------------------------------------------- propositions
print("\nSection IV  (two bounds, random instances)")


def debt_bound(n, B, k, rng):
    err = rng.integers(0, 2, n).astype(bool)
    A = rng.permutation(n)[:B]
    C = rng.permutation(n)
    outside = [int(x) for x in C if x not in set(A.tolist())]
    chosen, nonhit = [], 0
    for x in outside:
        if nonhit >= k or len(chosen) >= B:
            break
        chosen.append(x)
        if not err[x]:
            nonhit += 1
    for x in A:
        if len(chosen) >= B:
            break
        if x not in chosen:
            chosen.append(int(x))
    return int(err[A].sum()) - int(err[np.array(chosen[:B])].sum())


rng = np.random.default_rng(0)
worst = max((debt_bound(int(rng.integers(6, 20)), 8, k, rng) - k)
            for k in range(0, 9) for _ in range(500))
chk("max(shortfall - k) over 4,500 instances", float(worst), 0.0, 0.0)

viol = 0
for _ in range(20000):
    n = int(rng.integers(4, 14)); B = int(rng.integers(1, n))
    err = rng.integers(0, 2, n).astype(bool)
    A = rng.permutation(n)[:B]; C = rng.permutation(n)[:B]
    d = len(set(A.tolist()) - set(C.tolist()))
    if abs(int(err[A].sum()) - int(err[C].sum())) > d:
        viol += 1
chk("overlap-screen violations", viol, 0, 0)

print()
if fails:
    print(f"FAILURES ({len(fails)}): {fails}")
else:
    print("PASS: saved summary checks completed")
raise SystemExit(1 if fails else 0)
