"""Render the paired review-to-accuracy bridge from the recorded outcomes."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def render(path):
    primary = json.loads((HERE/"review_bridge/summary.json").read_text())
    linear = json.loads((HERE/"review_bridge_linear/summary.json").read_text())

    def fmt(x):
        return "$" + f"{x:+.2f}" + "$"

    def cell(x):
        return f"{fmt(x['mean'])} [{fmt(x['ci95'][0])}, {fmt(x['ci95'][1])}]"

    rows = []
    for key,train,test in (("n","CIFAR-10N","CIFAR-10.1"),
                           ("n","CIFAR-10N","CIFAR-10"),
                           ("h","CIFAR-10H","CIFAR-10.1")):
        a = primary[key]["endpoints"][test]["contrasts"]["difficulty_minus_random"]
        b = linear[key][test]["contrasts"]["difficulty_minus_random"]
        rows.append(" & ".join((train,test,fmt(primary[key]["difficulty_minus_random_yield"]),cell(a),cell(b))) + r" \\")
    path.write_text("\n".join(rows)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    render(p.parse_args().output)
