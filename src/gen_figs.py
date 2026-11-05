"""Figures for the recourse test. Regenerates from whatever runs exist."""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RES, PAPER = BASE / "results", BASE / "paper"
PAPER.mkdir(exist_ok=True)

BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.5, "font.family": "serif"})

CODE_LABEL = {
    "R1": "checking acct.", "R2": "credit history", "R3": "savings",
    "R4": "employment", "R5": "amount", "R6": "duration", "R7": "installment",
    "R8": "collateral", "R9": "other plans", "R10": "n credits",
    "R11": "no guarantor", "R12": "housing",
}


def load_s1(run):
    p = RES / f"{run}.json"
    if not p.exists():
        return None
    return json.load(open(p))["cases"]


def fig_arms():
    """Panel A: cited vs placebo vs joint by arm. Panel B: per-code flip rate."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.6),
                                   gridspec_kw={"width_ratios": [1, 1.35]})

    # --- Panel A: arms
    arms = []
    for run, lab in [("s1_haiku_n260_s3", "S1 haiku"), ("s1_haiku_n200_s4", "S1 haiku s4"),
                     ("s1_sonnet_n150_s3", "S1 sonnet"), ("s1_opus_n150_s3", "S1 opus"),
                     ("s1_gpt-4.1_n150_s3", "S1 GPT-4.1")]:
        cases = load_s1(run)
        if not cases:
            continue
        done = [c for c in cases.values() if "flip_cited" in c]
        if not done:
            continue
        fc = np.mean([c["flip_cited"] for c in done])
        fp = np.mean([c["flip_placebo"] for c in done if "flip_placebo" in c])
        n = len(done)
        arms.append((lab, fc, fp, n))
    for name, path in [("S2 blind", "s2_haiku_summary"),
                       ("S2 informed", "s2_haiku_informed_summary")]:
        p = RES / f"{path}.json"
        if p.exists():
            s = json.load(open(p))["summary"]
            arms.append((name, s["flip_cited"], s["flip_placebo"], s["n"]))

    y = np.arange(len(arms))
    h = 0.36
    for i, (lab, fc, fp, n) in enumerate(arms):
        ax1.barh(i - h/2, fc, height=h, color=BLUE,
                 label="cited repaired" if i == 0 else None)
        ax1.barh(i + h/2, fp, height=h, color=ORANGE,
                 label="placebo repaired" if i == 0 else None)
        ax1.annotate(f"{fc-fp:+.2f}", (max(fc, fp) + 0.02, i), va="center",
                     fontsize=6.5, color="#333")
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{a[0]}\n$n$={a[3]}" for a in arms], fontsize=6.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("fraction of denials that flip to approve")
    ax1.set_xlim(0, 1.0)
    ax1.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(length=2, width=0.5)
    ax1.set_title("A. Causal specificity by arm", fontsize=7.5, loc="left")

    # --- Panel B: per-code flip rate (pooled haiku S1)
    per = defaultdict(lambda: [0, 0])
    for run in ["s1_haiku_n260_s3", "s1_haiku_n200_s4"]:
        cases = load_s1(run) or {}
        for c in cases.values():
            if "flip_cited" in c and c["codes"]:
                per[c["codes"][0]][0] += 1
                per[c["codes"][0]][1] += c["flip_cited"]
    items = [(k, v[1] / v[0], v[0]) for k, v in per.items() if v[0] >= 5]
    items.sort(key=lambda x: x[1])
    ax2.barh([CODE_LABEL.get(k, k) for k, _, _ in items], [r for _, r, _ in items],
             color=GREEN, height=0.55)
    for i, (k, r, n) in enumerate(items):
        ax2.annotate(f"{r:.2f}  ($n$={n})", (r + 0.015, i), va="center", fontsize=6.5,
                     color="#333")
    ax2.set_xlabel("flip rate when cited as principal reason")
    ax2.set_xlim(0, 1.18)
    ax2.tick_params(labelsize=6.5, length=2, width=0.5)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("B. Pivotality varies by factor (S1 haiku, pooled)", fontsize=7.5, loc="left")

    fig.tight_layout(pad=0.4)
    fig.savefig(PAPER / "fig_arms.pdf")
    print("wrote fig_arms.pdf; arms:", [(a[0], round(a[1], 2), round(a[2], 2)) for a in arms])


if __name__ == "__main__":
    fig_arms()
