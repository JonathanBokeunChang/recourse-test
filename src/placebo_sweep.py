"""Full placebo sweep: repair EVERY adverse-but-uncited factor on every denial.

The single random placebo per denial answers "is the cited reason more pivotal
than an arbitrary adverse one?", but leaves the matched within-factor table
sparse: cited reasons concentrate on a few codes while placebo draws scatter.
Sweeping every uncited adverse factor gives, for each code, both a cited and a
not-cited flip rate at usable sample sizes -- the comparison that cannot be
explained by differences in repair strength.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_recourse import decide
from german import load, render, REASONS

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-per-case", type=int, default=5)
    args = ap.parse_args()

    path = RESULTS / f"{args.run}.json"
    state = json.load(open(path))
    cases, model = state["cases"], state["cfg"]["model"]
    rows = load()

    jobs = []
    for c in cases.values():
        if "flip_cited" not in c:
            continue
        sweep = c.setdefault("sweep", {})
        r = rows[c["idx"]]
        uncited = [k for k, (_, _, _, is_adv) in REASONS.items()
                   if k not in c["codes"] and is_adv(r) and k not in sweep]
        for code in uncited[: args.max_per_case]:
            jobs.append((str(c["idx"]), code))
    print(f"{args.run}: {len(jobs)} placebo repairs to sweep")

    def work(job):
        key, code = job
        r = rows[cases[key]["idx"]]
        d, _, _ = decide(render(REASONS[code][2](r)), model)
        return key, code, (d == "APPROVE")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (key, code, flip) in enumerate(ex.map(work, jobs)):
            cases[key].setdefault("sweep", {})[code] = flip
            if i % 25 == 0:
                json.dump(state, open(path, "w"), indent=1)
                print(f"sweep {i}/{len(jobs)}", flush=True)
    json.dump(state, open(path, "w"), indent=1)

    import numpy as np
    from collections import defaultdict
    cited, uncited = defaultdict(list), defaultdict(list)
    for c in cases.values():
        if "flip_cited" in c:
            cited[c["codes"][0]].append(c["flip_cited"])
        for code, f in c.get("sweep", {}).items():
            uncited[code].append(f)
    print("\ncode   cited (n)      not-cited (n)   gap")
    for code in sorted(set(cited) | set(uncited)):
        cv, uv = cited.get(code, []), uncited.get(code, [])
        if len(cv) >= 5 and len(uv) >= 5:
            print(f"{code:5s} {np.mean(cv):.2f} ({len(cv):3d})   "
                  f"{np.mean(uv):.2f} ({len(uv):3d})   {np.mean(cv)-np.mean(uv):+.2f}")


if __name__ == "__main__":
    main()
