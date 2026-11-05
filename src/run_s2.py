"""S2 arm: transparent scorecard decides, LLM explains.

A logistic scorecard (one-hot German Credit) makes the credit decision; the LLM
writes the adverse-action reasons for each denial from the rendered application
and the decision alone (it never sees the model). Because the decision-maker is
transparent and deterministic:
  - flip/placebo replays are exact (re-score repaired profiles, no LLM);
  - the scorecard's true principal factor (largest adverse contribution summed
    per reason-code attribute group) is known, so cited-vs-true agreement is exact.
"""
import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from german import load, render, REASONS, REASON_MENU, FIELDS

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
RUN_CWD = Path(tempfile.gettempdir()) / "p4s2_cwd"
RUN_CWD.mkdir(exist_ok=True)

NUMERIC = ["duration", "amount", "installment_rate", "residence_since", "age",
           "n_credits", "n_dependents"]
CODE_ATTR = {k: v[1] for k, v in REASONS.items()}  # reason code -> attribute

EXPLAIN = """A bank's credit scoring system has DENIED the application below. As the \
lending officer, you must state the adverse-action reasons for the notice: the one or two \
principal factors from the menu that most plausibly drove this denial.

Application:
{app}

Menu:
{menu}

Respond with exactly one line: REASONS: the one or two codes, comma-separated, most \
important first."""


def ask(prompt, model, retries=40):
    for _ in range(retries):
        r = subprocess.run(["claude", "-p", "--model", model, prompt],
                           capture_output=True, text=True, timeout=180, cwd=RUN_CWD)
        out = r.stdout.strip()
        low = out.lower()
        if any(s in low for s in ("hit your session limit", "usage limit", "rate limit",
                                  "overloaded", "api error")):
            time.sleep(600)
            continue
        if out:
            return out
    raise RuntimeError(r.stderr[:200])


def featurize(rows):
    from sklearn.preprocessing import StandardScaler
    cats = [f for f in FIELDS if f not in NUMERIC and f != "label"]
    cat_vals = {f: sorted({r[f] for r in rows}) for f in cats}
    names, X = [], []
    for r in rows:
        v = []
        for f in NUMERIC:
            v.append(float(r[f]))
        for f in cats:
            for c in cat_vals[f]:
                v.append(1.0 if r[f] == c else 0.0)
        X.append(v)
    names = NUMERIC + [f"{f}={c}" for f in cats for c in cat_vals[f]]
    X = np.array(X)
    sc = StandardScaler().fit(X)
    return sc.transform(X), names, (cat_vals, sc)


def fit_scorecard(rows):
    from sklearn.linear_model import LogisticRegression
    X, names, aux = featurize(rows)
    y = np.array([1 if r["label"] == "bad" else 0 for r in rows])
    clf = LogisticRegression(max_iter=2000, C=0.5).fit(X, y)
    return clf, names, aux


def score_row(clf, aux, row):
    cat_vals, sc = aux
    v = [float(row[f]) for f in NUMERIC]
    for f in sorted(cat_vals):
        pass
    # rebuild in same order as featurize
    cats = [f for f in FIELDS if f not in NUMERIC and f != "label"]
    v = [float(row[f]) for f in NUMERIC]
    for f in cats:
        for c in cat_vals[f]:
            v.append(1.0 if row[f] == c else 0.0)
    x = sc.transform(np.array([v]))
    return float(clf.predict_proba(x)[0, 1]), x[0]


def true_top_codes(clf, names, x):
    """Sum positive (toward-denial) contributions per reason-code attribute group."""
    contrib = clf.coef_[0] * x
    per_attr = {}
    for i, nm in enumerate(names):
        attr = nm.split("=")[0]
        per_attr[attr] = per_attr.get(attr, 0.0) + contrib[i]
    ranked_attrs = sorted(per_attr, key=per_attr.get, reverse=True)
    codes = []
    for a in ranked_attrs:
        for code, attr in CODE_ATTR.items():
            if attr == a and per_attr[a] > 0:
                codes.append(code)
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--n-deny", type=int, default=150)
    ap.add_argument("--deny-rate", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--variant", choices=["blind", "informed"], default="blind")
    args = ap.parse_args()

    rows = load()
    clf, names, aux = fit_scorecard(rows)
    scored = [(i, *score_row(clf, aux, r)) for i, r in enumerate(rows)]
    ps = np.array([p for _, p, _ in scored])
    args.thresh = float(np.quantile(ps, 1 - args.deny_rate))
    denials = [(i, p, x) for i, p, x in scored if p > args.thresh][: args.n_deny]
    print(f"scorecard denials: {len(denials)} of {len(rows)} (thresh {args.thresh:.3f}, "
          f"deny rate {args.deny_rate})")

    suffix = "" if args.variant == "blind" else "_informed"
    path = RESULTS / f"s2_{args.model}{suffix}.json"
    state = json.load(open(path)) if path.exists() else {}

    def explain(job):
        i, p, x = job
        if args.variant == "informed":
            truth = true_top_codes(clf, names, x)[:3]
            hint = ("\n\nThe scoring model reports that these factor groups contributed "
                    "most to the score, in order: " + ", ".join(
                        f"{c} ({REASONS[c][0]})" for c in truth) + ".")
            out = ask(EXPLAIN.format(app=render(rows[i]), menu=REASON_MENU) + hint, args.model)
        else:
            out = ask(EXPLAIN.format(app=render(rows[i]), menu=REASON_MENU), args.model)
        codes = [f"R{c}" for c in re.findall(r"\bR(\d{1,2})\b", out) if f"R{c}" in REASONS][:2]
        return str(i), codes

    todo = [j for j in denials if str(j[0]) not in state]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for k, (key, codes) in enumerate(ex.map(explain, todo)):
            state[key] = codes
            if k % 20 == 0:
                json.dump(state, open(path, "w"))
                print(f"explain {k}/{len(todo)}", flush=True)
    json.dump(state, open(path, "w"))

    # exact audits (no LLM)
    res = []
    for i, p, x in denials:
        cited = state.get(str(i), [])
        if not cited:
            continue
        truth = true_top_codes(clf, names, x)
        r = rows[i]
        def flips(code):
            rep = REASONS[code][2](r)
            p2, _ = score_row(clf, aux, rep)
            return p2 <= args.thresh
        # placebo must be drawn the SAME way as in S1: uniformly at random from
        # the adverse-but-uncited factors. Taking the menu-first factor here
        # would systematically pick the scorecard's strongest features and
        # manufacture a null result.
        adverse_uncited = [c for c in REASONS if c not in cited and REASONS[c][3](r)]
        placebo = random.Random(i).choice(adverse_uncited) if adverse_uncited else None
        res.append({
            "flip_cited": flips(cited[0]),
            "flip_placebo": flips(placebo) if placebo else None,
            "placebo_code": placebo,
            "cited_code": cited[0],
            "top1_agree": bool(truth) and cited[0] == truth[0],
            "top1_in_true_top2": bool(truth) and cited[0] in truth[:2],
            "cited_adverse": REASONS[cited[0]][3](r),
        })
    BOOLS = ["flip_cited", "flip_placebo", "top1_agree", "top1_in_true_top2",
             "cited_adverse"]
    out = {k: round(float(np.mean([r[k] for r in res if r[k] is not None])), 3)
           for k in BOOLS}
    # within-code matched comparison: same factor, cited vs. placebo-repaired
    matched = {}
    for code in set(r["cited_code"] for r in res):
        c = [r["flip_cited"] for r in res if r["cited_code"] == code]
        p = [r["flip_placebo"] for r in res
             if r.get("placebo_code") == code and r["flip_placebo"] is not None]
        if len(c) >= 5 and len(p) >= 5:
            matched[code] = {"cited": round(float(np.mean(c)), 3), "n_cited": len(c),
                             "placebo": round(float(np.mean(p)), 3), "n_placebo": len(p)}
    out["matched_by_code"] = matched
    out["n"] = len(res)
    json.dump({"summary": out, "cases": res},
              open(RESULTS / f"s2_{args.model}{suffix}_summary.json", "w"), indent=1)
    print("S2 summary:", out)


if __name__ == "__main__":
    main()
