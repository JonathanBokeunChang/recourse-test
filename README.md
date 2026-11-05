# The Recourse Test

Code and replay records for **"The Recourse Test: Do LLM Adverse-Action Explanations Survive Repair-and-Replay?"** by Jonathan Chang, accepted at IAFM '26: Workshop on Interpretability & Alignment of Financial Models at ICAIF '26 (Milan, November 2026). The workshop is non-archival.

## What the test does

When a lender denies credit, U.S. law (ECOA / Regulation B) requires a statement of the principal reasons. The recourse test checks whether an LLM's stated reason actually drives its decision:

1. An LLM underwriter reads a loan application (a German Credit profile written out as a natural-language file), decides, and cites reason codes from a fixed menu of 12.
2. For each denial we **repair** the cited factor (set it to a favorable value) and replay the application.
3. We compare against repairing an adverse factor the notice did **not** cite (placebo), replaying the unchanged file (noise floor), repairing up to five uncited factors one at a time (sweep), repairing both cited factors (joint), and re-rendering the file in a shuffled field order (stability).

A second arm (S2) asks an LLM to explain the decisions of a transparent logistic-regression scorecard, either blind or given the scorecard's factor contributions.

## Repository layout

| Path | Contents |
|---|---|
| `data/german.data`, `data/german.doc` | Statlog (German Credit) dataset, Hofmann (1994), UCI Machine Learning Repository, CC BY 4.0 |
| `src/german.py` | Dataset loading, natural-language rendering, reason-code menu, adversity predicates and repair operators |
| `src/run_recourse.py` | S1 experiment: underwriter decisions, cited / placebo / no-repair / joint replays |
| `src/placebo_sweep.py`, `src/control_backfill.py`, `src/joint_backfill.py` | Within-denial sweep and backfilled control arms |
| `src/run_s2.py` | S2 experiment: scorecard plus blind and informed LLM narrators |
| `src/stats.py`, `src/rigor.py`, `src/matched_analysis.py`, `src/percode.py`, `src/reviewer_baselines.py` | Statistics (McNemar, Wilcoxon, Wilson intervals), controls, per-code tables, reviewer-requested baselines (oracle, precision/recall, fixed-order explainer, S2 ceiling) |
| `src/gen_results.py`, `src/gen_figs.py` | Generate the paper's tables and figures into `paper/` |
| `results/*.json` | Saved replay records for every run (one record per application) |
| `results/*.log` | Run logs |

## Reproducing the paper's numbers (no API calls)

All tables and statistics are computed from the saved records in `results/`:

```bash
pip install -r requirements.txt
cd src
python rigor.py && python matched_analysis.py && python percode.py \
  && python reviewer_baselines.py && python gen_results.py && python gen_figs.py
```

Generated LaTeX tables and macros are written to `paper/`.

## Re-running the experiments (makes model calls)

- Claude underwriters (`haiku`, `sonnet`, `opus`) were called through the Claude Code CLI (`claude -p --model <alias>`) between 29 September and 1 October 2026 at provider-default decoding (the CLI adds its default system prompt). Exact model snapshot identifiers were not recorded, so re-runs may use newer models behind the same aliases.
- GPT-4.1 was called through the OpenAI API (`src/gpt_client.py`, `max_completion_tokens=300`, default temperature). The key is read from `~/.config/icaif/openai_key` and is never stored in the repository.
- One sample per replay. Application samples are drawn with `random.Random(seed).sample(range(1000), n)` (seed 3 for the primary run of 260, seed 4 for the second run of 200).

## Notes

- The LLM prompt shows 16 application fields, including age; it excludes sex/personal status, foreign-worker status, years at residence and number of dependents. The S2 scorecard uses all 20 attributes.
- This is a research audit protocol, not a legal compliance test.

## License

Code: MIT (see `LICENSE`). The German Credit data is redistributed under its original CC BY 4.0 license.
