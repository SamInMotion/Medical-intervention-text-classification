"""Power analysis for the topic-stratified BoW gap.

Uses observed per-fold variance to compute the minimum detectable effect
(MDE) at each topic, and reports it at two fold counts.

PART 1, the canonical single-run design (n=5 folds per topic). The question
this answers is what the canonical Cohen evaluation design could have
detected, and that design is one 5-fold run. This is the analysis behind the
paper's power table.

PART 2, the pooled multi-run fold count (n=35 per topic). The paper's main
results tables use this fold count, so the same quantity computed there is
reported for contrast. It assumes the 35 per-fold differences are
independent, which they are not: folds within a rerun share four fifths of
their training data and all reruns share the corpus. The two parts bracket
the answer; neither is the effective sample size.

Part 2 also emits the per-fold mean and standard deviation of the multi-run
differences, which are the source of the design-sensitivity table's SD
column.

CHANGES FROM THE PREVIOUS VERSION
  P1  MDE is now reported from the exact noncentral-t distribution, not the
      normal approximation. The exact value was already computable here via
      t_inflation_factor(); it was never applied to the reported number.
      The normal approximation is retained as a secondary column.
  P2  Removed "so the values below are conservative". Whether an understated
      MDE is conservative depends on the claim it supports, and the sentence
      asserted it unconditionally.
  P3  t_inflation_factor() silently returned None at some fold counts. The
      bracket [1e-9, 3*sd] drives the noncentrality past the range where
      scipy's nct is stable (it failed at n=25 and n=70, worked at n=5 and
      n=35), and the caller filtered the Nones out without noticing. Replaced
      with nct.sf and an adaptive bracket, verified at n = 5, 7, 25, 35, 70
      and 105.
  P4  The detectability comparison now uses the exact MDE.
  P5  Part 2 added.
  P6  Removed a named personal reference and a machine-specific absolute
      path from the module docstring and search paths. This file ships in the
      anonymised review repository.

Reads bow_stats_results.json for Part 1 and
outputs/bow_{topic}_multirun_summary.json for Part 2.

Outputs:
    paper_experiments/outputs/power_analysis.md

Usage:
    python paper_experiments/power_analysis.py
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path("paper_experiments/outputs")
ALPHA = 0.05
POWER_TARGET = 0.80
N_BOOT = 10_000
SEED = 42

EXPERT_MODE = "title_abstract_mesh"
AUTO_MODE = "auto_mesh"


def find_bow_stats():
    candidates = [
        Path("bow_stats_results.json"),
        Path("paper_experiments/bow_stats_results.json"),
    ]
    env = os.environ.get("COHEN_BOW_STATS")
    if env:
        candidates.append(Path(env))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def per_fold_diffs_for_topic(stats, topic):
    """Try a few shapes to extract per-fold expert-vs-auto diffs for a topic."""
    if not stats:
        return None

    if "per_topic" in stats and isinstance(stats["per_topic"], list):
        for entry in stats["per_topic"]:
            if entry.get("topic") == topic:
                if "diffs" in entry:
                    return list(entry["diffs"])
                if "per_run" in entry:
                    pooled = []
                    for r in entry["per_run"]:
                        if "diffs" in r:
                            pooled.extend(r["diffs"])
                    if pooled:
                        return pooled

    if topic in stats and isinstance(stats[topic], dict):
        entry = stats[topic]
        if "diffs" in entry:
            return list(entry["diffs"])
        if "per_run" in entry:
            pooled = []
            for r in entry["per_run"]:
                if isinstance(r, dict) and "diffs" in r:
                    pooled.extend(r["diffs"])
            if pooled:
                return pooled
        if "expert_wss_per_fold" in entry and "auto_wss_per_fold" in entry:
            ex = entry["expert_wss_per_fold"]
            au = entry["auto_wss_per_fold"]
            if len(ex) == len(au):
                return [a - b for a, b in zip(ex, au)]

    return None


def multirun_diffs_for_topic(topic_slug):
    """Per-fold expert-minus-auto differences pooled across the multi-run set.

    Reads outputs/bow_{slug}_multirun_summary.json, structure
    d['runs'][i]['modes'][mode] -> list of per-fold WSS@95 values.
    """
    path = Path(f"outputs/bow_{topic_slug}_multirun_summary.json")
    if not path.exists():
        return None, path
    d = json.loads(path.read_text(encoding="utf-8"))
    runs = [r["modes"] for r in d["runs"]]
    if not all(m in runs[0] for m in (EXPERT_MODE, AUTO_MODE)):
        return None, path
    diffs = [e - a for m in runs
             for e, a in zip(m[EXPERT_MODE], m[AUTO_MODE])]
    return diffs, path


def t_inflation_factor(n, alpha=ALPHA, power=POWER_TARGET):
    """Exact noncentral-t MDE divided by the normal-approximation MDE.

    Scale-free: depends only on n, alpha and power, not on the observed SD.
    Uses nct.sf rather than 1 - nct.cdf, and brackets the root by expansion
    rather than at a fixed multiple of sd, because a fixed bracket pushes the
    noncentrality into a range where scipy's nct returns NaN at some n.
    """
    df = n - 1
    if df < 1:
        return None
    from scipy.optimize import brentq
    from scipy.stats import norm, nct, t as student_t

    crit = student_t.ppf(1 - alpha / 2, df)

    def attained(delta):
        ncp = delta * math.sqrt(n)
        return nct.sf(crit, df, ncp) + nct.cdf(-crit, df, ncp)

    hi = 0.05
    while attained(hi) < power and hi < 50:
        hi *= 1.5
    if attained(hi) < power:
        return None
    exact = float(brentq(lambda d: attained(d) - power, 1e-9, hi, xtol=1e-12))
    approx = (norm.ppf(1 - alpha / 2) + norm.ppf(power)) / math.sqrt(n)
    if approx == 0:
        return None
    return exact / approx


def mde_from_diffs(diffs, alpha=ALPHA, power=POWER_TARGET):
    """MDE for a one-sample t-style test on paired differences.

    Reports the exact noncentral-t value as `mde`, and the normal
    approximation as `mde_normal` for comparison. The approximation
    understates the MDE, materially so at small n.
    """
    arr = np.asarray(diffs, dtype=float)
    n = len(arr)
    if n < 2:
        return None
    sd = float(arr.std(ddof=1))
    from scipy.stats import norm
    approx = (norm.ppf(1 - alpha / 2) + norm.ppf(power)) * sd / math.sqrt(n)
    infl = t_inflation_factor(n, alpha, power)
    if infl is None:
        return None
    return {
        "n": n,
        "mean": float(arr.mean()),
        "sd": sd,
        "se": sd / math.sqrt(n),
        "mde_normal": float(approx),
        "mde": float(approx * infl),
        "inflation": float(infl),
    }


def bootstrap_ci(diffs, n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
    arr = np.asarray(diffs, dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    return (
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


TOPICS = [
    ("Statins", "statins", 2744),
    ("Opiods", "opiods", 1772),
    ("ADHD", "adhd", 803),
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stats_path = find_bow_stats()
    if not stats_path:
        print(
            "bow_stats_results.json not found. Expected at ./ or "
            "./paper_experiments/, or set COHEN_BOW_STATS.",
            file=sys.stderr,
        )
        sys.exit(1)

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    print(f"Loaded single-run BoW stats from: {stats_path}")

    # ---- Part 1, canonical single-run design -----------------------------
    rows = []
    for topic, _slug, n_total in TOPICS:
        diffs = per_fold_diffs_for_topic(stats, topic)
        if not diffs:
            print(f"[warn] No per-fold diffs found for {topic}", file=sys.stderr)
            continue
        m = mde_from_diffs(diffs)
        if m is None:
            print(f"[warn] MDE not computable for {topic}", file=sys.stderr)
            continue
        ci_lo, ci_hi = bootstrap_ci(diffs)
        rows.append({"topic": topic, "n_total": n_total, "ci_lo": ci_lo,
                     "ci_hi": ci_hi, **m})

    if not rows:
        print("No topic data could be extracted.", file=sys.stderr)
        sys.exit(1)

    # ---- Part 2, pooled multi-run fold count -----------------------------
    multi = []
    for topic, slug, n_total in TOPICS:
        diffs, path = multirun_diffs_for_topic(slug)
        if diffs is None:
            print(f"[warn] multi-run summary unusable: {path}", file=sys.stderr)
            continue
        m = mde_from_diffs(diffs)
        if m is None:
            continue
        multi.append({"topic": topic, "n_total": n_total, "n_runs": len(diffs) // 5,
                      **m})

    statins_row = next((r for r in rows if r["topic"] == "Statins"), None)

    md = []
    md.append("# Power analysis: minimum detectable effect by topic\n")

    # Part 1
    md.append("## Part 1. Canonical single-run design\n")
    n_values = sorted({r["n"] for r in rows})
    n_desc = str(n_values[0]) if len(n_values) == 1 else "/".join(map(str, n_values))
    md.append(
        f"Per-fold expert-vs-auto WSS@95 differences from `{stats_path.name}`, "
        f"the canonical single-run 5-fold analysis. MDE computed at "
        f"alpha={ALPHA} (two-sided), power={POWER_TARGET:.2f}, from the exact "
        f"noncentral-t distribution, on n={n_desc} fold values per topic.\n"
    )
    infl = rows[0]["inflation"]
    md.append(
        f"At n={n_desc} the normal approximation understates the MDE by about "
        f"{(1 - 1 / infl) * 100:.0f}% of the exact value (the exact value is "
        f"{infl:.3f} times the approximation). The approximation is shown for "
        f"comparison only.\n"
    )
    md.append("| Topic | n_total | n_folds | Observed mean | 95% CI | SD | SE | "
              "MDE exact (80% power) | MDE normal approx |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['topic']} | {r['n_total']} | {r['n']} | {r['mean']:+.4f} | "
            f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['sd']:.4f} | "
            f"{r['se']:.4f} | {r['mde']:.4f} | {r['mde_normal']:.4f} |"
        )
    md.append("")

    if statins_row:
        md.append("### What this answers\n")
        md.append(
            "The observed Statins effect at this design is "
            f"**{statins_row['mean']:+.4f}**, against an exact MDE of "
            f"{statins_row['mde']:.4f}. Compare that effect size against the "
            "MDE at each smaller topic:\n"
        )
        for r in rows:
            if r["topic"] == "Statins":
                continue
            detectable = statins_row["mean"] >= r["mde"]
            verdict = ("WOULD have been detected" if detectable
                       else "would NOT have been detected")
            md.append(
                f"- **{r['topic']}** (exact MDE = {r['mde']:.4f}): a "
                f"Statins-sized effect ({statins_row['mean']:+.4f}) {verdict} "
                f"at this topic's variance and fold count."
            )
        md.append("")
        md.append(
            "If a Statins-sized effect would have been detectable at "
            "Opioids/ADHD given their variance, the absence of a gap at those "
            "topics is informative about the effect. If the MDE is larger "
            "than the Statins effect, this design alone does not distinguish "
            "an absent gap from an undetected one.\n"
        )

    # Part 2
    if multi:
        md.append("## Part 2. Pooled multi-run fold count\n")
        md.append(
            "The same quantity at the fold count used in the main results "
            "tables. This treats the pooled per-fold differences as "
            "independent observations, which they are not: folds within a "
            "rerun share four fifths of their training data and all reruns "
            "share the corpus. The effective sample size lies between Part 1 "
            "and Part 2 and is not determined by this design. The mean and SD "
            "columns are the source of the design-sensitivity table's "
            "multi-run rows.\n"
        )
        md.append("| Topic | n_runs | n_folds | Mean | SD | MDE exact (80% power) |")
        md.append("|---|---|---|---|---|---|")
        for r in multi:
            md.append(
                f"| {r['topic']} | {r['n_runs']} | {r['n']} | {r['mean']:+.4f} | "
                f"{r['sd']:.4f} | {r['mde']:.4f} |"
            )
        md.append("")
        s_multi = next((r for r in multi if r["topic"] == "Statins"), None)
        if s_multi:
            md.append(
                f"Statins effect at this fold count: **{s_multi['mean']:+.4f}**. "
                "Report both parts together; neither on its own bounds the "
                "cross-topic null.\n"
            )
    else:
        md.append("## Part 2. Pooled multi-run fold count\n")
        md.append(
            "Not computed: no readable `outputs/bow_{topic}_multirun_summary.json`.\n"
        )

    out_path = OUTPUT_DIR / "power_analysis.md"
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")
    print("")
    print("Part 1, single-run design:")
    for r in rows:
        print(
            f"  {r['topic']:8s} n={r['n']:3d}  mean={r['mean']:+.4f}  "
            f"sd={r['sd']:.4f}  MDE exact={r['mde']:.4f}  "
            f"(normal {r['mde_normal']:.4f})"
        )
    if multi:
        print("Part 2, pooled multi-run fold count:")
        for r in multi:
            print(
                f"  {r['topic']:8s} n={r['n']:3d}  mean={r['mean']:+.4f}  "
                f"sd={r['sd']:.4f}  MDE exact={r['mde']:.4f}"
            )


if __name__ == "__main__":
    main()
