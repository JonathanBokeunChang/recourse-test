"""Significance tests for the recourse test: two-proportion comparisons."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats as sps

RES = Path(__file__).resolve().parent.parent / "results"

def arm(path):
    cases = json.load(open(path))["cases"]
    done = [c for c in cases.values() if "flip_cited" in c]
    fc = [c["flip_cited"] for c in done]
    fp = [c["flip_placebo"] for c in done if "flip_placebo" in c]
    fj = [c["flip_joint"] for c in done if "flip_joint" in c]
    return fc, fp, fj

for run in ["s1_haiku_n260_s3", "s1_haiku_n200_s4", "s1_sonnet_n150_s3"]:
    p = RES / f"{run}.json"
    if not p.exists():
        continue
    fc, fp, fj = arm(p)
    if not fc or not fp:
        continue
    tab = np.array([[sum(fc), len(fc)-sum(fc)], [sum(fp), len(fp)-sum(fp)]])
    odds, pval = sps.fisher_exact(tab)
    print(f"{run}: flip_cited={np.mean(fc):.3f}(n={len(fc)}) flip_placebo={np.mean(fp):.3f}(n={len(fp)}) "
          f"joint={np.mean(fj):.3f}(n={len(fj)}) Fisher p={pval:.2e}")
