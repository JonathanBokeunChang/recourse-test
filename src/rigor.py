"""Corrections demanded by review: within-denial matching, chance baselines,
the S2 ceiling, and the paired tests the main table was missing.

Three things the first analysis got wrong:
  * The matched table compared cited-vs-uncited ACROSS denials, so it controlled
    repair strength but not denial difficulty. Denials that cite savings happen
    to be ones where almost any repair flips the outcome. The within-denial
    contrast -- same denial, cited factor vs each uncited factor -- removes both.
  * S2 top-factor agreement was reported with no chance floor. Always naming the
    scorecard's most common top factor scores 55%, so the blind narrator's 39%
    is BELOW chance, and the informed narrator's 83% is partly copy fidelity.
  * The main table reported Fisher's exact on paired data after the code's own
    docstring said that test is wrong.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps

BASE = Path(__file__).resolve().parent.parent
RES, PAPER = BASE / "results", BASE / "paper"
sys.path.insert(0, str(Path(__file__).parent))
from german import REASONS

# short labels: the full Reg B wording overflows a single column
SHORT = {
    "R1": "Checking balance", "R2": "Credit history", "R3": "Savings",
    "R4": "Employment", "R5": "Loan amount", "R6": "Loan duration",
    "R7": "Installment burden", "R8": "Collateral", "R9": "Other plans",
    "R10": "Existing credits", "R11": "No guarantor", "R12": "Housing",
}
LABEL = {k: SHORT.get(k, v[0]) for k, v in REASONS.items()}


def within_denial(run="s1_haiku_n260_s3"):
    """For each denial, the cited factor's flip vs the mean flip of that SAME
    denial's uncited factors. Controls denial difficulty as well as repair
    strength."""
    cases = json.load(open(RES / f"{run}.json"))["cases"]
    per = defaultdict(lambda: {"c": [], "u": []})
    pairs = []
    for c in cases.values():
        if "flip_cited" not in c or not c["codes"] or not c.get("sweep"):
            continue
        code = c["codes"][0]
        u = list(c["sweep"].values())
        if not u:
            continue
        per[code]["c"].append(float(c["flip_cited"]))
        per[code]["u"].append(float(np.mean(u)))
        pairs.append(float(c["flip_cited"]) - float(np.mean(u)))
    rows = []
    for code, v in per.items():
        if len(v["c"]) >= 10:
            d = np.array(v["c"]) - np.array(v["u"])
            t = sps.wilcoxon(d)[1] if np.any(d != 0) else 1.0
            rows.append((code, float(np.mean(v["c"])), float(np.mean(v["u"])),
                         float(np.mean(d)), len(v["c"]), t))
    return rows, np.array(pairs)


def s2_stats(tag):
    p = RES / f"s2_haiku{tag}_summary.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    cases = d["cases"]
    fc = np.array([c["flip_cited"] for c in cases if c["flip_placebo"] is not None])
    fp = np.array([c["flip_placebo"] for c in cases if c["flip_placebo"] is not None])
    b = int(np.sum(fc & ~fp))
    c_ = int(np.sum(~fc & fp))
    p_mc = sps.binomtest(b, b + c_, 0.5).pvalue if (b + c_) else 1.0
    delta = float(fc.mean() - fp.mean())
    se = float(np.sqrt((fc.astype(float) - fp.astype(float)).var(ddof=1) / len(fc)))
    return {"n": len(fc), "delta": delta, "ci": (delta - 1.96 * se, delta + 1.96 * se),
            "p": p_mc, "top1": d["summary"]["top1_agree"],
            "top2": d["summary"]["top1_in_true_top2"]}


def main():
    rows, pairs = within_denial()
    obs = float(pairs.mean())
    p_all = sps.wilcoxon(pairs)[1]
    print(f"within-denial gap, pooled: {obs:+.3f}  n={len(pairs)}  Wilcoxon p={p_all:.2e}")
    for code, cv, uv, d, n, t in sorted(rows, key=lambda r: -r[3]):
        print(f"  {code:4s} {LABEL[code][:34]:34s} cited {cv:.2f} uncited {uv:.2f} "
              f"within {d:+.2f} (n={n}, p={t:.1e})")

    blind, informed = s2_stats(""), s2_stats("_informed")
    # chance floor for top-factor agreement: always name the modal true factor
    cases = json.load(open(RES / "s2_haiku_summary.json"))["cases"]
    # recover the modal true top factor from agreement structure is not possible;
    # recompute directly
    sys.path.insert(0, str(Path(__file__).parent))
    from run_s2 import load, fit_scorecard, score_row, true_top_codes
    rws = load()
    clf, names, aux = fit_scorecard(rws)
    scored = [(i, *score_row(clf, aux, r)) for i, r in enumerate(rws)]
    ps = np.array([p for _, p, _ in scored])
    thr = float(np.quantile(ps, 0.8))
    tops = [true_top_codes(clf, names, x)[0] for i, p, x in scored if p > thr][:150]
    from collections import Counter
    modal, modal_n = Counter(tops).most_common(1)[0]
    floor = modal_n / len(tops)
    print(f"\nS2 top-factor chance floor: always say {modal} -> {floor:.3f} (n={len(tops)})")
    for name, st in [("blind", blind), ("informed", informed)]:
        if st:
            print(f"  {name:8s} delta={st['delta']:+.3f} CI[{st['ci'][0]:+.3f},{st['ci'][1]:+.3f}] "
                  f"p={st['p']:.4f} top1={st['top1']:.3f} top2={st['top2']:.3f}")

    m = ["% AUTO-GENERATED by rigor.py",
         f"\\renewcommand{{\\WithinGap}}{{{obs:+.2f}}}",
         f"\\renewcommand{{\\WithinN}}{{{len(pairs)}}}",
         f"\\renewcommand{{\\SBfloor}}{{{100*floor:.0f}\\%}}",
         f"\\renewcommand{{\\SBmodal}}{{{modal}}}"]
    if blind:
        m += [f"\\renewcommand{{\\SBp}}{{{blind['p']:.2f}}}",
              f"\\renewcommand{{\\SBci}}{{[{blind['ci'][0]:+.2f}, {blind['ci'][1]:+.2f}]}}"]
    if informed:
        m += [f"\\renewcommand{{\\SIp}}{{{informed['p']:.4f}}}",
              f"\\renewcommand{{\\SIci}}{{[{informed['ci'][0]:+.2f}, {informed['ci'][1]:+.2f}]}}"]
    body = "\n".join(
        f"{LABEL[code]} & {cv:.2f} & {uv:.2f} & ${d:+.2f}$ & {n} \\\\"
        for code, cv, uv, d, n, t in sorted(rows, key=lambda r: -r[3]))
    m += ["\\renewcommand{\\WithinTable}{%",
          "\\begin{tabular}{lcccc}", "\\toprule",
          "Cited factor & cited & uncited, same denial & $\\Delta$ & $n$ \\\\",
          "\\midrule", body, "\\bottomrule", "\\end{tabular}}"]
    (PAPER / "rigor_gen.tex").write_text("\n".join(m) + "\n")
    print("\nwrote rigor_gen.tex")


if __name__ == "__main__":
    main()
