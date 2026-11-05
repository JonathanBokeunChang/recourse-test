"""Backfill the joint-repair condition (repair ALL cited reasons at once) on
completed recourse runs."""
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
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    path = RESULTS / f"{args.run}.json"
    state = json.load(open(path))
    cases = state["cases"]
    model = state["cfg"]["model"]
    rows = load()

    todo = [c for c in cases.values()
            if "flip_cited" in c and len(c["codes"]) > 1 and "flip_joint" not in c]
    print(f"{args.run}: backfilling joint repair for {len(todo)} two-code denials")

    def work(c):
        r = rows[c["idx"]]
        for code in c["codes"]:
            r = REASONS[code][2](r)
        d, _, _ = decide(render(r), model)
        return str(c["idx"]), (d == "APPROVE")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (key, flip) in enumerate(ex.map(work, todo)):
            cases[key]["flip_joint"] = flip
            if i % 15 == 0:
                json.dump(state, open(path, "w"), indent=1)
                print(f"joint {i}/{len(todo)}", flush=True)
    json.dump(state, open(path, "w"), indent=1)

    import numpy as np
    done = [c for c in cases.values() if "flip_joint" in c]
    one = [c for c in cases.values() if "flip_cited" in c and len(c["codes"]) == 1]
    print(f"flip(joint, 2-code)={np.mean([c['flip_joint'] for c in done]):.3f} n={len(done)}")
    print(f"(1-code denials for reference: flip(cited)={np.mean([c['flip_cited'] for c in one]):.3f} n={len(one)})")


if __name__ == "__main__":
    main()
