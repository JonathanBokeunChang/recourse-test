"""No-repair control: replay each denial with the application UNCHANGED.

Any flip here is pure decision noise (resubmission variance), which sets the
floor against which the cited and placebo flip rates must be read. Without it,
a placebo flip rate of ~13% cannot be distinguished from "the model sometimes
changes its mind."
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_recourse import decide
from german import load, render

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    path = RESULTS / f"{args.run}.json"
    state = json.load(open(path))
    cases, model = state["cases"], state["cfg"]["model"]
    rows = load()

    todo = [c for c in cases.values() if "flip_cited" in c and "flip_noop" not in c]
    print(f"{args.run}: no-repair control on {len(todo)} denials")

    def work(c):
        d, _, _ = decide(render(rows[c["idx"]]), model)
        return str(c["idx"]), (d == "APPROVE")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (key, flip) in enumerate(ex.map(work, todo)):
            cases[key]["flip_noop"] = flip
            if i % 20 == 0:
                json.dump(state, open(path, "w"), indent=1)
                print(f"noop {i}/{len(todo)}", flush=True)
    json.dump(state, open(path, "w"), indent=1)

    import numpy as np
    done = [c for c in cases.values() if "flip_noop" in c]
    print(f"flip(no repair) = {np.mean([c['flip_noop'] for c in done]):.3f}  n={len(done)}")
    print(f"  vs flip(placebo) = {np.mean([c['flip_placebo'] for c in done if 'flip_placebo' in c]):.3f}")
    print(f"  vs flip(cited)   = {np.mean([c['flip_cited'] for c in done]):.3f}")


if __name__ == "__main__":
    main()
