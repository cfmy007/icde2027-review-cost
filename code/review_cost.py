"""Replay fixed source choices against human reference labels and account for every check."""
import argparse
import csv
import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

from threshold import allocation, intervals_for_counts


P = json.loads(Path(__file__).with_name("review_cost_protocol.json").read_text())
TAIL = .025


def cumulative(errors, rank):
    return np.r_[0, np.cumsum(errors[rank], dtype=int)]


def signed_debt(random, hard, budget):
    difference = random[budget] - hard[budget]
    if difference > 0:
        return int(np.searchsorted(hard, random[budget]) - budget)
    if difference < 0:
        return int(budget - np.searchsorted(random, hard[budget]))
    return 0


def exclusive(random, hard):
    return np.setdiff1d(random, hard, assume_unique=True), np.setdiff1d(
        hard, random, assume_unique=True)


def bounds(size, sample):
    return intervals_for_counts(size, sample, tuple(range(sample + 1)), TAIL)


@lru_cache(maxsize=20000)
def exact_power(sizes, hits, draws, threshold):
    """Probability that the exact one-sided lower bound reaches threshold."""
    if hits[0] - hits[1] < threshold and sum(draws) == sum(sizes):
        return 0.
    low, _ = bounds(sizes[0], draws[0])
    _, high = bounds(sizes[1], draws[1])
    cut = np.searchsorted(high, low - threshold, side="right")
    probabilities = [
        hypergeom.pmf(np.arange(m + 1), size, k, m) if size else np.ones(1)
        for size, k, m in zip(sizes, hits, draws)]
    result = float(probabilities[0] @ np.r_[0, probabilities[1].cumsum()][cut])
    return min(1., max(0., result))


@lru_cache(maxsize=4000)
def exact_critical(sizes, draws, threshold):
    """Worst-case finite-population null for the estimated hit-count difference."""
    nr, nh = sizes
    mr, mh = draws
    weight_r, weight_h = nr * max(1, mh), nh * max(1, mr)
    xr, xh = np.arange(mr + 1, dtype=np.int64), np.arange(mh + 1, dtype=np.int64)
    random_scores, hard_scores = weight_r*xr, weight_h*xh
    grid = np.unique((random_scores[:, None] - hard_scores[None, :]).ravel())
    # Under the null, for each hard total, the largest allowed random total
    # maximizes the upper tail. The null has no other nuisance parameter.
    kh = np.arange(max(0, 1-threshold), nh + 1)
    kr = np.minimum(nr, kh + threshold - 1)
    if not len(kh):
        return int(grid[0])
    p_random = hypergeom.pmf(xr, nr, kr[:, None], mr) if nr else np.ones((len(kr), 1))
    p_hard = hypergeom.pmf(xh, nh, kh[:, None], mh) if nh else np.ones((len(kh), 1))
    cdf_hard = np.c_[np.zeros(len(kh)), p_hard.cumsum(axis=1)]
    left, right = 0, len(grid)
    while left < right:
        index = (left + right) // 2
        eligible = np.searchsorted(hard_scores, random_scores - grid[index], side="right")
        worst = np.max(np.sum(p_random * cdf_hard[:, eligible], axis=1))
        if worst <= .05 - 1e-12:
            right = index
        else:
            left = index + 1
    return int(grid[left]) if left < len(grid) else int(grid[-1] + 1)


def sharper_power(sizes, hits, draws, threshold):
    """Exact fixed-cost power with nuisance maximization rather than two bounds."""
    nr, nh = sizes
    mr, mh = draws
    if sum(draws) == sum(sizes):
        return float(hits[0] - hits[1] >= threshold)
    if not sum(draws) or nr < threshold:
        return 0.
    critical = exact_critical(sizes, draws, threshold)
    xr, xh = np.arange(mr + 1, dtype=np.int64), np.arange(mh + 1, dtype=np.int64)
    random_scores, hard_scores = nr * max(1, mh)*xr, nh * max(1, mr)*xh
    cut = np.searchsorted(hard_scores, random_scores - critical, side="right")
    pr = hypergeom.pmf(xr, nr, hits[0], mr) if nr else np.ones(1)
    ph = hypergeom.pmf(xh, nh, hits[1], mh) if nh else np.ones(1)
    return min(1., max(0., float(pr @ np.r_[0, ph.cumsum()][cut])))


def measurement(kind, random, hard, errors, budget, minimum, extra):
    hard_limit = budget if kind == "repairs" else budget + extra - 1
    threshold = minimum if kind == "repairs" else 1
    sides = exclusive(random[:budget], hard[:hard_limit])
    sizes = tuple(map(len, sides))
    hits = tuple(int(errors[side].sum()) for side in sides)
    assert hits[0] - hits[1] == int(errors[random[:budget]].sum()) - int(
        errors[hard[:hard_limit]].sum())
    costs = sorted({min(sum(sizes), round(budget * f))
                    for f in P["measurement_budget_fractions"]} | {sum(sizes)})
    positive_costs = [cost for cost in costs if cost]
    allocations = dict(zip(positive_costs, allocation(sizes, positive_costs))) if positive_costs else {}
    for cost in costs:
        draws = tuple(map(int, allocations.get(cost, (0, 0))))
        yield dict(test=kind, cost=cost, audit_fraction=cost / budget, random_only=sizes[0],
                   difficulty_only=sizes[1], random_only_hits=hits[0], difficulty_only_hits=hits[1],
                   random_draws=draws[0], difficulty_draws=draws[1],
                   material=int(hits[0] - hits[1] >= threshold),
                   power=exact_power(sizes, hits, draws, threshold),
                   exact_tail_power=sharper_power(sizes, hits, draws, threshold))


def interleave(random, hard, budget):
    seen = set()
    for first, second in zip(random[:budget], hard[:budget]):
        for item in (first, second):
            seen.add(item)
            if len(seen) == budget:
                return np.fromiter(seen, dtype=int)
    raise ValueError("Rankings must contain at least B distinct items")


def audit_strata(random, hard, budget, m):
    while m:
        a, b = exclusive(random[:budget - m], hard[:budget - m])
        if len(a) >= m:
            return a, b, m
        m -= 1
    return np.empty(0, dtype=int), np.empty(0, dtype=int), 0


def deployment(random, hard, errors, budget, rng):
    a, b, m = audit_strata(random, hard, budget, min(budget // 20, 50))
    h = cumulative(errors, hard)
    r = cumulative(errors, random)
    mixed = int(errors[interleave(random, hard, budget)].sum())
    draws = P["audit_repetitions"]
    if m:
        # Draw independent uniform samples in the two strata. The observed hit
        # count is sufficient for both the policy decision and its credited yield.
        x, y = [rng.hypergeometric(int(errors[s].sum()), len(s) - int(errors[s].sum()),
                                   m, size=draws) for s in (a, b)]
        low, high = bounds(len(a), m)
        switch = low[x] - high[y] + y - x > 0
        audited_r, audited_h = r[budget - m] + y, h[budget - m] + x
        chosen = np.where(switch, audited_r, audited_h)
        mistaken = switch & (audited_r <= audited_h)
    else:
        x = y = switch = mistaken = np.zeros(draws, dtype=int)
        chosen = np.full(draws, h[budget])
    for i in range(draws):
        yield dict(repetition=i, checks=budget, audit=2*m, random_audit_hits=int(x[i]),
                   difficulty_audit_hits=int(y[i]), switched=int(switch[i]),
                   mistaken_switch=int(mistaken[i]),
                   yield_audited=int(chosen[i]), yield_random=int(r[budget]),
                   yield_difficulty=int(h[budget]), yield_interleave=mixed)


def exact_deployment(random, hard, errors, budget):
    r, h = cumulative(errors, random), cumulative(errors, hard)
    mixed = int(errors[interleave(random, hard, budget)].sum())
    for fraction in P["deployment_curve_fractions"]:
        a, b, m = audit_strata(random, hard, budget, math.floor(fraction*budget/2))
        if m:
            x = np.arange(m + 1, dtype=int)
            pr, ph = [hypergeom.pmf(x, len(s), int(errors[s].sum()), m) for s in (a, b)]
            joint = pr[:, None] * ph[None, :]
            low, high = bounds(len(a), m)
            switch = (low-x)[:, None] > (high-x)[None, :]
            chosen = np.where(switch, r[budget-m]+x[None, :], h[budget-m]+x[:, None])
            expected = float((joint*chosen).sum())
            switch_rate = float((joint*switch).sum())
        else:
            expected, switch_rate = float(h[budget]), 0.
        yield dict(audit_fraction_limit=fraction, audit=2*m, expected_yield=expected,
                   switch_probability=switch_rate, yield_random=int(r[budget]),
                   yield_difficulty=int(h[budget]), yield_interleave=mixed)


def row_metrics(random, hard, errors, budget, minimum, extra):
    r, h = cumulative(errors, random), cumulative(errors, hard)
    assert r[-1] == h[-1] == errors.sum()
    debt = signed_debt(r, h, budget)
    if debt > 0:
        assert h[budget + debt - 1] < r[budget] <= h[budget + debt]
    elif debt < 0:
        assert r[budget - debt - 1] < h[budget] <= r[budget - debt]
    d = len(exclusive(random[:budget], hard[:budget])[0])
    gap = int(r[budget] - h[budget])
    assert abs(gap) <= d
    return dict(random_hits=int(r[budget]), difficulty_hits=int(h[budget]), gap=gap,
                gap_per_10000=gap*10000/len(errors), signed_extra_checks=debt,
                extra_over_budget=debt/budget, exclusive_each=d, material_fixes=minimum,
                material_checks=extra, ruled_out_by_overlap=int(d < minimum),
                difficulty_debt_ruled_out=int(np.isin(random[:budget], hard[:budget+extra-1]).all()),
                random_debt_ruled_out=int(np.isin(hard[:budget], random[:budget+extra-1]).all()),
                reference_prevalence=float(errors.mean()))


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_check():
    small_r = np.array([0, 1, 2, 3, 4])
    small_h = np.array([2, 3, 4, 1, 0])
    errors = np.array([1, 1, 0, 0, 0], dtype=bool)
    assert row_metrics(small_r, small_h, errors, 2, 1, 1)["signed_extra_checks"] == 3
    assert row_metrics(small_h, small_r, errors, 2, 1, 1)["signed_extra_checks"] == -3
    for a in range(5):
        for b in range(6):
            for m in range(1, 5):
                low, _ = bounds(4, m)
                _, high = bounds(5, m)
                direct = sum(hypergeom.pmf(i, 4, a, m)*hypergeom.pmf(j, 5, b, m)
                             for i in range(m + 1) for j in range(m + 1) if low[i]-high[j] >= 1)
                assert np.isclose(exact_power((4, 5), (a, b), (m, m), 1), direct)
                if a <= b:
                    assert direct <= .05
                    assert sharper_power((4, 5), (a, b), (m, m), 1) <= .05


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    self_check()
    if args.second:
        with np.load(args.second) as data:
            second = dict(data)
        assert set(second) == set(P["second_reference"])
    else:
        second = {}
    metrics, power, deployed, ties, curve = [], [], [], [], []
    datasets = sorted(d for d in args.inputs.iterdir() if (d / "target.npz").exists())
    assert len(datasets) == 17
    for i, directory in enumerate(datasets):
        name = directory.name
        with np.load(directory / "target.npz") as data:
            ranks, names = data["rank"], data["names"]
            original = data["errors"].astype(bool)
        with (directory / "paired.csv").open(encoding="utf-8", newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["menu"] == "expanded"]
        counts = Counter(float(row["budget"]) for row in rows)
        assert counts == dict.fromkeys(P["budgets"], 10)
        for j, row in enumerate(rows):
            frac = float(row["budget"])
            k, l = int(row["random"]), int(row["difficulty"])
            r, h = ranks[k], ranks[l]
            n = len(original)
            b = round(frac*n)
            minimum = math.ceil(n*P["material_fixes_per_10000"]/10000)
            extra = math.ceil(b*P["material_extra_fraction"])
            shared = dict(dataset=name, seed=int(row["seed"]), budget=frac, n=n, checks=b,
                          random_detector=names[k], difficulty_detector=names[l])
            outcomes = row_metrics(r, h, original, b, minimum, extra)
            assert outcomes["random_hits"] == int(row["random_hits"])
            assert outcomes["difficulty_hits"] == int(row["difficulty_hits"])
            metrics.append(dict(**shared, reference=1, **outcomes))
            references = {1: original}
            if name in second:
                other = second[name].astype(bool)
                metrics.append(dict(**shared, reference=2, **row_metrics(r, h, other, b, minimum, extra)))
                references[2] = other
            source_r, source_h = [np.array(json.loads(row[col]))
                                  for col in ("random_source", "difficulty_source")]
            tied_r = np.flatnonzero(source_r == source_r.max())
            tied_h = np.flatnonzero(source_h == source_h.max())
            assert k == tied_r[0] and l == tied_h[0]
            base = {col: value for col, value in shared.items()
                    if col not in ("random_detector", "difficulty_detector")}
            for reference, truth in references.items():
                for candidate_r in tied_r:
                    for candidate_h in tied_h:
                        ties.append(dict(**base, reference=reference, random_detector=names[candidate_r],
                                         difficulty_detector=names[candidate_h],
                                         **row_metrics(ranks[candidate_r], ranks[candidate_h], truth, b,
                                                       minimum, extra)))
            if frac != P["primary_budget"] or args.metrics_only:
                continue
            for kind in ("repairs", "extra_checks"):
                for reference, truth in references.items():
                    power.extend(dict(**shared, reference=reference, **result)
                                 for result in measurement(kind, r, h, truth, b, minimum, extra))
            rng = np.random.default_rng(np.random.SeedSequence([P["seed"], i, j]))
            for result in deployment(r, h, original, b, rng):
                assert b == result["checks"]
                deployed.append(dict(**shared, **{k: v for k, v in result.items() if k != "checks"}))
            for reference, truth in references.items():
                curve.extend(dict(**shared, reference=reference, **result)
                             for result in exact_deployment(r, h, truth, b))
        print(name, "complete", flush=True)
    write(args.out / "paired_costs.csv", metrics)
    write(args.out / "source_tie_sensitivity.csv", ties)
    if power:
        write(args.out / "fixed_audit_power.csv", power)
        write(args.out / "paid_policy_replay.csv", deployed)
        write(args.out / "paid_policy_exact.csv", curve)


if __name__ == "__main__":
    main()
