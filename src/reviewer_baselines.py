"""Reviewer-1 baselines, computed from saved run records only (no model calls).

  (a) ORACLE        share of audited denials where SOME single repair flipped the
                    decision (cited top repair, the placebo replay, or any swept
                    uncited repair).
  (b) PRECISION /   cited principal reasons vs pivotal factors, where "pivotal"
      RECALL        = the factor's single repair flipped the decision, among the
                    factors that were actually repaired.
  (c) STATIC        a fixed-order explainer that cites the first adverse factor in
      EXPLAINER     a global order (citation frequency), scored with the replay
                    outcomes that already exist for that factor; coverage reported.
  (d) S2 SCORECARD  the deterministic scorecard explaining itself: repair its own
      SELF-EXPLAIN  top contributing factor vs a placebo, exact re-scoring.

Writes results/reviewer_baselines.json and paper/revmacros_gen.tex.
S1 numbers use the primary run (s1_haiku_n260_s3) unless noted: it is the only
run carrying the uncited sweep and the no-repair control.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats as sps

BASE = Path(__file__).resolve().parent.parent
RES, PAPER = BASE / "results", BASE / "paper"
sys.path.insert(0, str(Path(__file__).parent))
from german import load, REASONS  # noqa: E402
from run_s2 import fit_scorecard, score_row, true_top_codes  # noqa: E402

PRIMARY = "s1_haiku_n260_s3"
S1_RUNS = ["s1_haiku_n260_s3", "s1_haiku_n200_s4", "s1_sonnet_n150_s3",
           "s1_opus_n150_s3", "s1_gpt-4.1_n150_s3"]
MENU = list(REASONS)  # R1..R12 in menu order
ROWS = load()


def adverse(code, r):
    return bool(REASONS[code][3](r))


def audited(run):
    cases = json.load(open(RES / f"{run}.json"))["cases"]
    return [c for c in cases.values() if "flip_cited" in c and c["codes"]]


def outcomes(c):
    """code -> flip for every factor that received a SINGLE-factor repair replay.
    The cited top code uses flip_cited; uncited codes use the sweep, falling back
    to the placebo replay when the placebo code was not swept. The second cited
    code is never repaired alone (only jointly), so it has no entry."""
    o = {c["codes"][0]: bool(c["flip_cited"])}
    for k, v in (c.get("sweep") or {}).items():
        o[k] = bool(v)
    pk = c.get("placebo_code")
    if pk and pk not in o and c.get("flip_placebo") is not None:
        o[pk] = bool(c["flip_placebo"])
    return o


def rate(v):
    return float(np.mean(v)) if len(v) else None


def wilson(k, n):
    if not n:
        return (None, None)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(m - h), float(m + h))


def paired(a, b):
    """Paired difference mean(a)-mean(b), normal CI, McNemar exact p."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    d = a.astype(float) - b.astype(float)
    bb, cc = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = sps.binomtest(bb, bb + cc, 0.5).pvalue if bb + cc else 1.0
    se = float(np.sqrt(d.var(ddof=1) / len(d))) if len(d) > 1 else 0.0
    return {"delta": float(d.mean()), "ci": [float(d.mean() - 1.96 * se),
                                             float(d.mean() + 1.96 * se)],
            "mcnemar_b": bb, "mcnemar_c": cc, "mcnemar_p": float(p), "n": int(len(d))}


# --------------------------------------------------------------------- (a)
def oracle():
    out = {}
    den = audited(PRIMARY)
    noop = rate([c["flip_noop"] for c in den if "flip_noop" in c])
    any_main, any_all, k_rep, complete = [], [], [], []
    for c in den:
        o = outcomes(c)
        any_main.append(any(o.values()))
        # sensitivity: also count the placebo replay when it duplicates a swept code
        any_all.append(any(o.values()) or bool(c.get("flip_placebo")))
        k_rep.append(len(o))
        r = ROWS[c["idx"]]
        unc = [k for k in MENU if k not in c["codes"] and adverse(k, r)]
        complete.append(len(c["codes"]) == 1 and len(unc) <= 5)
    k_rep = np.array(k_rep)
    noise_any = float(np.mean(1 - (1 - noop) ** k_rep))
    n = len(den)
    out["primary"] = {
        "run": PRIMARY, "n": n,
        "oracle": rate(any_main), "k": int(sum(any_main)), "wilson": wilson(sum(any_main), n),
        "oracle_incl_dup_placebo": rate(any_all), "k_incl_dup_placebo": int(sum(any_all)),
        "flip_cited": rate([c["flip_cited"] for c in den]),
        "mean_single_repairs_per_denial": float(k_rep.mean()),
        "total_single_repairs": int(k_rep.sum()),
        "noop_rate": noop,
        "expected_any_flip_under_pure_noise": noise_any,
        "n_denials_exhaustively_covered": int(sum(complete)),
        "oracle_on_exhaustive": rate([a for a, f in zip(any_main, complete) if f]),
        "n_oracle_but_not_cited": int(sum(1 for a, c in zip(any_main, den)
                                          if a and not c["flip_cited"])),
    }
    # other runs: only cited + one placebo replay exist -> weaker lower bound
    for run in S1_RUNS:
        d = audited(run)
        v = [bool(c["flip_cited"]) or bool(c.get("flip_placebo")) for c in d]
        out.setdefault("cited_or_placebo_by_run", {})[run] = {
            "n": len(d), "rate": rate(v), "k": int(sum(v))}
    return out


# --------------------------------------------------------------------- (b)
def prec_recall():
    den = audited(PRIMARY)
    cited_piv = sum(1 for c in den if c["flip_cited"])
    unc_rep = unc_piv = 0
    den_any = den_c1 = 0
    unc_piv_list = []
    for c in den:
        o = outcomes(c)
        u = {k: v for k, v in o.items() if k != c["codes"][0]}
        unc_rep += len(u)
        unc_piv += sum(u.values())
        unc_piv_list.extend(k for k, v in u.items() if v)
        if any(o.values()):
            den_any += 1
            den_c1 += int(c["flip_cited"])
    n = len(den)
    n_c2 = sum(1 for c in den if len(c["codes"]) == 2)
    tot_piv = cited_piv + unc_piv
    return {
        "run": PRIMARY, "n_denials": n,
        "precision_principal": cited_piv / n, "precision_num": cited_piv, "precision_den": n,
        "n_second_codes_never_singly_repaired": n_c2,
        "repaired_factor_instances": n + unc_rep, "repaired_uncited": unc_rep,
        "pivotal_instances": tot_piv, "pivotal_cited": cited_piv, "pivotal_uncited": unc_piv,
        "recall_micro": cited_piv / tot_piv if tot_piv else None,
        "recall_denial_level": den_c1 / den_any if den_any else None,
        "recall_denial_num": den_c1, "recall_denial_den": den_any,
        "uncited_pivotal_by_code": dict(Counter(unc_piv_list).most_common()),
        "uncited_flip_rate": unc_piv / unc_rep if unc_rep else None,
    }


# --------------------------------------------------------------------- (c)
def global_order(run, principal_only=True):
    cnt = Counter()
    for c in audited(run):
        for k in (c["codes"][:1] if principal_only else c["codes"]):
            cnt[k] += 1
    # by frequency, then menu order for ties and never-cited codes
    return sorted(MENU, key=lambda k: (-cnt.get(k, 0), MENU.index(k))), dict(cnt)


def static_explainer(order, label):
    den = audited(PRIMARY)
    rec = []
    for c in den:
        r = ROWS[c["idx"]]
        adv = [k for k in order if adverse(k, r)]
        if not adv:
            rec.append({"idx": c["idx"], "static": None, "src": "no_adverse"})
            continue
        s = adv[0]
        o = outcomes(c)
        if s == c["codes"][0]:
            src = "llm_cited_top"
        elif s in (c.get("sweep") or {}):
            src = "sweep"
        elif s == c.get("placebo_code"):
            src = "placebo"
        elif len(c["codes"]) > 1 and s == c["codes"][1]:
            src = "missing_llm_second_code"
        else:
            src = "missing_not_swept"
        flip = o.get(s) if src in ("llm_cited_top", "sweep", "placebo") else None
        others = [v for k, v in o.items() if k != s]
        rec.append({"idx": c["idx"], "static": s, "src": src, "flip": flip,
                    "llm_flip": bool(c["flip_cited"]),
                    "placebo_code": c.get("placebo_code"),
                    "llm_placebo": c.get("flip_placebo"),
                    "others_mean": float(np.mean(others)) if others else None,
                    "llm_others_mean": (float(np.mean([v for k, v in o.items()
                                                        if k != c["codes"][0]]))
                                        if len(o) > 1 else None)})
    n = len(rec)
    cov = [x for x in rec if x.get("flip") is not None]
    out = {"label": label, "order": order, "n": n, "n_covered": len(cov),
           "coverage": len(cov) / n,
           "source_counts": dict(Counter(x["src"] for x in rec)),
           "agrees_with_llm_top": sum(1 for x in rec if x["src"] == "llm_cited_top"),
           "flip_static": rate([x["flip"] for x in cov]),
           "flip_llm_same_subset": rate([x["llm_flip"] for x in cov])}
    # vs the LLM's placebo replay on the covered denials (placebo must differ from
    # the static code, else the "placebo" is the static repair itself)
    pv = [x for x in cov if x["llm_placebo"] is not None and x["placebo_code"] != x["static"]]
    if pv:
        out["vs_placebo"] = paired([x["flip"] for x in pv], [x["llm_placebo"] for x in pv])
        out["vs_placebo"]["flip_static"] = rate([x["flip"] for x in pv])
        out["vs_placebo"]["flip_placebo"] = rate([bool(x["llm_placebo"]) for x in pv])
        out["llm_vs_placebo_same_subset"] = paired([x["llm_flip"] for x in pv],
                                                   [x["llm_placebo"] for x in pv])
    # within-denial: static code vs mean of that denial's other single repairs
    wd = [x for x in cov if x["others_mean"] is not None]
    if wd:
        d = np.array([float(x["flip"]) - x["others_mean"] for x in wd])
        out["within_denial_gap"] = float(d.mean())
        out["within_denial_n"] = len(wd)
        out["within_denial_p"] = float(sps.wilcoxon(d)[1]) if np.any(d != 0) else 1.0
    return out


# --------------------------------------------------------------------- (d)
def s2_self():
    rows = ROWS
    clf, names, aux = fit_scorecard(rows)
    scored = [(i, *score_row(clf, aux, r)) for i, r in enumerate(rows)]
    ps = np.array([p for _, p, _ in scored])
    thr = float(np.quantile(ps, 0.8))
    above = [(i, p, x) for i, p, x in scored if p > thr]
    den = above[:150]

    def p_after(i, code):
        return score_row(clf, aux, REASONS[code][2](rows[i]))[0]

    # validation: re-derive the stored blind-narrator audit from the refit
    stored = json.load(open(RES / "s2_haiku_summary.json"))["cases"]
    cites = json.load(open(RES / "s2_haiku.json"))
    recon = []
    for i, p, x in den:
        cited = cites.get(str(i), [])
        if not cited:
            continue
        r = rows[i]
        au = [c for c in REASONS if c not in cited and adverse(c, r)]
        pk = random.Random(i).choice(au) if au else None
        recon.append({"flip_cited": p_after(i, cited[0]) <= thr,
                      "flip_placebo": (p_after(i, pk) <= thr) if pk else None,
                      "placebo_code": pk, "cited_code": cited[0]})
    keys = ["flip_cited", "flip_placebo", "placebo_code", "cited_code"]
    match = len(recon) == len(stored) and all(
        a[k] == b[k] for a, b in zip(recon, stored) for k in keys)

    res = []
    for i, p, x in den:
        r = rows[i]
        truth = true_top_codes(clf, names, x)
        if not truth:
            res.append({"idx": i, "top": None})
            continue
        top = truth[0]
        cited2 = truth[:2]
        au2 = [c for c in REASONS if c not in cited2 and adverse(c, r)]
        au1 = [c for c in REASONS if c != top and adverse(c, r)]
        pk2 = random.Random(i).choice(au2) if au2 else None
        pk1 = random.Random(i).choice(au1) if au1 else None
        pa = {c: p_after(i, c) for c in REASONS}
        adv_codes = [c for c in REASONS if adverse(c, r)]
        best_any = min(pa, key=pa.get)
        best_adv = min(adv_codes, key=pa.get) if adv_codes else None
        res.append({
            "idx": i, "p": p, "top": top, "top_adverse": adverse(top, r),
            "flip_top": pa[top] <= thr,
            "placebo2": pk2, "flip_placebo2": (pa[pk2] <= thr) if pk2 else None,
            "placebo1": pk1, "flip_placebo1": (pa[pk1] <= thr) if pk1 else None,
            "others_adv_mean": (float(np.mean([pa[c] <= thr for c in adv_codes if c != top]))
                                if [c for c in adv_codes if c != top] else None),
            "best_any": best_any, "best_adv": best_adv,
            "oracle_any": any(v <= thr for v in pa.values()),
            "oracle_adv": any(pa[c] <= thr for c in adv_codes),
            "pivotal_adv": [c for c in adv_codes if pa[c] <= thr],
        })
    ok = [z for z in res if z["top"] is not None]
    p2 = [z for z in ok if z["flip_placebo2"] is not None]
    p1 = [z for z in ok if z["flip_placebo1"] is not None]
    wd = [z for z in ok if z["others_adv_mean"] is not None]
    d_wd = np.array([float(z["flip_top"]) - z["others_adv_mean"] for z in wd])

    # exact precision/recall for the blind LLM narrator on S2 (every factor repairable)
    piv_cited = piv_tot = 0
    llm_p = []
    for i, p, x in den:
        cited = cites.get(str(i), [])
        z = next(q for q in res if q["idx"] == i)
        if not cited or z["top"] is None:
            continue
        llm_p.append(p_after(i, cited[0]) <= thr)
        piv_tot += len(z["pivotal_adv"])
        piv_cited += sum(1 for c in z["pivotal_adv"] if c in cited)

    return {
        "threshold": thr, "n_above_threshold": len(above), "n_denials": len(den),
        "denials_are_first_150_by_index": True,
        "refit_reproduces_stored_blind_audit": bool(match), "n_recon": len(recon),
        "n_with_top": len(ok),
        "top_code_counts": dict(Counter(z["top"] for z in ok).most_common()),
        "top_is_adverse_share": rate([z["top_adverse"] for z in ok]),
        "flip_top": rate([z["flip_top"] for z in ok]), "n_flip_top": len(ok),
        "vs_placebo_excl_top2": {**paired([z["flip_top"] for z in p2],
                                          [z["flip_placebo2"] for z in p2]),
                                 "flip_top": rate([z["flip_top"] for z in p2]),
                                 "flip_placebo": rate([z["flip_placebo2"] for z in p2])},
        "vs_placebo_excl_top1": {**paired([z["flip_top"] for z in p1],
                                          [z["flip_placebo1"] for z in p1]),
                                 "flip_top": rate([z["flip_top"] for z in p1]),
                                 "flip_placebo": rate([z["flip_placebo1"] for z in p1])},
        "within_denial_gap": float(d_wd.mean()), "within_denial_n": len(wd),
        "oracle_any_code": rate([z["oracle_any"] for z in ok]),
        "oracle_adverse_codes": rate([z["oracle_adv"] for z in ok]),
        "top_equals_recourse_optimal_any": rate([z["top"] == z["best_any"] for z in ok]),
        "top_equals_recourse_optimal_adverse": rate([z["top"] == z["best_adv"] for z in ok]),
        "recourse_optimal_any_counts": dict(Counter(z["best_any"] for z in ok).most_common()),
        "recourse_optimal_adverse_counts": dict(Counter(z["best_adv"] for z in ok).most_common()),
        "llm_blind_precision": rate(llm_p), "llm_blind_n": len(llm_p),
        "llm_blind_recall_micro": piv_cited / piv_tot if piv_tot else None,
        "llm_blind_recall_num": piv_cited, "llm_blind_recall_den": piv_tot,
    }


# --------------------------------------------------------------------- output
def pct(x, nd=0):
    return "--" if x is None else f"{100 * x:.{nd}f}\\%"


def sgn(x):
    return "--" if x is None else f"{x:+.2f}"


def main():
    A, B = oracle(), prec_recall()
    ord_in, cnt_in = global_order(PRIMARY, True)
    ord_all, _ = global_order(PRIMARY, False)
    ord_oos, cnt_oos = global_order("s1_haiku_n200_s4", True)
    C = {"order_in_sample_principal_counts": cnt_in,
         "order_out_of_sample_principal_counts": cnt_oos,
         "in_sample": static_explainer(ord_in, "principal-citation frequency, primary run"),
         "in_sample_all_positions": static_explainer(ord_all, "all-position frequency, primary run"),
         "out_of_sample": static_explainer(ord_oos, "principal-citation frequency, seed-4 run")}
    D = s2_self()
    allres = {"oracle": A, "precision_recall": B, "static": C, "s2_scorecard_self": D}
    json.dump(allres, open(RES / "reviewer_baselines.json", "w"), indent=1, default=float)

    a, s, d = A["primary"], C["in_sample"], D["vs_placebo_excl_top2"]
    sv = s.get("vs_placebo", {})
    m = {
        "RevOracle": pct(a["oracle"]), "RevOracleK": a["k"], "RevOracleN": a["n"],
        "RevOracleRepairs": a["total_single_repairs"],
        "RevOracleNoise": pct(a["expected_any_flip_under_pure_noise"]),
        "RevOracleNotCited": a["n_oracle_but_not_cited"],
        "RevPrecision": pct(B["precision_principal"]), "RevPrecisionK": B["precision_num"],
        "RevPrecisionN": B["precision_den"],
        "RevRecall": pct(B["recall_micro"]), "RevRecallK": B["pivotal_cited"],
        "RevRecallN": B["pivotal_instances"],
        "RevRecallDenial": pct(B["recall_denial_level"]), "RevRecallDenialK": B["recall_denial_num"],
        "RevRecallDenialN": B["recall_denial_den"],
        "RevUncitedRepaired": B["repaired_uncited"], "RevUncitedPivotal": B["pivotal_uncited"],
        "RevStaticCoverage": pct(s["coverage"]), "RevStaticCovK": s["n_covered"],
        "RevStaticN": s["n"], "RevStaticAgree": s["agrees_with_llm_top"],
        "RevStaticFlip": pct(s["flip_static"]), "RevStaticLLMFlip": pct(s["flip_llm_same_subset"]),
        "RevStaticDelta": sgn(sv.get("delta")), "RevStaticDeltaN": sv.get("n", "--"),
        "RevStaticLLMDelta": sgn(s.get("llm_vs_placebo_same_subset", {}).get("delta")),
        "RevStaticWithin": sgn(s.get("within_denial_gap")),
        "RevStaticOOSFlip": pct(C["out_of_sample"]["flip_static"]),
        "RevStaticOOSCoverage": pct(C["out_of_sample"]["coverage"]),
        "RevSTwoN": d["n"], "RevSTwoTopFlip": f"{d['flip_top']:.2f}",
        "RevSTwoPlacebo": f"{d['flip_placebo']:.2f}", "RevSTwoDelta": sgn(d["delta"]),
        "RevSTwoCI": f"[{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]",
        "RevSTwoP": f"{d['mcnemar_p']:.2g}",
        "RevSTwoOracle": pct(D["oracle_adverse_codes"]),
        "RevSTwoTopIsOptimal": pct(D["top_equals_recourse_optimal_adverse"]),
        "RevSTwoWithin": sgn(D["within_denial_gap"]),
    }
    lines = ["% AUTO-GENERATED by reviewer_baselines.py -- do not edit"]
    for k, v in m.items():
        lines.append(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}")
    (PAPER / "revmacros_gen.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(allres, indent=1, default=float))


if __name__ == "__main__":
    main()
