"""Power analysis for the topic-stratified BoW gap.

Uses observed per-fold variance from the multi-run BoW data to compute the
minimum detectable effect (MDE) at each topic size, given the current
fold count. Answers Christer's Q2/Q3 from a complementary angle: even
without the matched-n subsampling experiment, we can ask whether the
observed Statins effect size would have been detectable at Opioids/ADHD
sample sizes if it existed there.

Reads bow_stats_results.json (searched in three likely locations) and
extracts per-fold expert-vs-auto WSS@95 diffs per topic.

Outputs:
    paper_experiments/outputs/power_analysis.md  (table + narrative for §5.2)

Usage:
    python paper_experiments/power_analysis.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path("paper_experiments/outputs")
ALPHA = 0.05
POWER_TARGET = 0.80
N_BOOT = 10_000
SEED = 42


def find_bow_stats():
    for candidate in [
        Path("bow_stats_results.json"),
        Path("paper_experiments/bow_stats_results.json"),
        Path("/g/My Drive/cohen_bert_run/bow_stats_results.json"),
    ]:
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


def mde_from_diffs(diffs, alpha=ALPHA, power=POWER_TARGET):
    """Minimum detectable effect for a one-sample t-style test on paired diffs.

    Approximation: MDE = (z_{1-a/2} + z_{power}) * SD / sqrt(n)

    For n=35 folds, the normal-approximation MDE is within a few percent of
    the t-distribution-corrected version. Documented in the output.
    """
    arr = np.asarray(diffs, dtype=float)
    n = len(arr)
    if n < 2:
        return None
    sd = float(arr.std(ddof=1))
    from scipy.stats import norm
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    mde = (z_a + z_b) * sd / math.sqrt(n)
    return {
        "n": n,
        "mean": float(arr.mean()),
        "sd": sd,
        "se": sd / math.sqrt(n),
        "mde": float(mde),
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stats_path = find_bow_stats()
    if not stats_path:
        print(
            "bow_stats_results.json not found in any of the expected locations:\n"
            "  ./bow_stats_results.json\n"
            "  ./paper_experiments/bow_stats_results.json\n"
            "  /g/My Drive/cohen_bert_run/bow_stats_results.json\n"
            "Copy the file to one of those paths, or modify find_bow_stats().",
            file=sys.stderr,
        )
        sys.exit(1)

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    print(f"Loaded BoW stats from: {stats_path}")

    topics = [
        ("Statins", 2744),
        ("Opiods", 1772),
        ("ADHD", 803),
    ]
    rows = []
    for topic, n_total in topics:
        diffs = per_fold_diffs_for_topic(stats, topic)
        if not diffs:
            print(f"[warn] No per-fold diffs found for {topic}", file=sys.stderr)
            continue
        m = mde_from_diffs(diffs)
        if m is None:
            continue
        ci_lo, ci_hi = bootstrap_ci(diffs)
        rows.append({
            "topic": topic,
            "n_total": n_total,
            "n_folds": m["n"],
            "mean": m["mean"],
            "sd": m["sd"],
            "se": m["se"],
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "mde": m["mde"],
        })

    if not rows:
        print(
            "No topic data could be extracted. bow_stats_results.json may use "
            "a schema not handled by per_fold_diffs_for_topic. Inspect it and "
            "extend the function.",
            file=sys.stderr,
        )
        sys.exit(1)

    statins_row = next((r for r in rows if r["topic"] == "Statins"), None)

    md = []
    md.append("# Power analysis: minimum detectable effect by topic\n")
    md.append(
        "Per-fold expert-vs-auto WSS@95 distributions from the BoW multi-run "
        "data give us topic-specific variance estimates. MDE computed at "
        "α=0.05 (two-sided), power=0.80, normal approximation. With n=35 "
        "fold values per topic the t-correction shifts MDE by under 4%.\n"
    )
    md.append("| Topic | n_total | n_folds | Observed mean | 95% CI | SD | SE | MDE (80% power) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['topic']} | {r['n_total']} | {r['n_folds']} | "
            f"{r['mean']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
            f"{r['sd']:.4f} | {r['se']:.4f} | {r['mde']:.4f} |"
        )
    md.append("")

    if statins_row:
        md.append("## What this answers")
        md.append("")
        md.append(
            "The observed Statins effect is "
            f"**{statins_row['mean']:+.4f}**. Compare this against the MDE at "
            "each smaller topic:"
        )
        md.append("")
        for r in rows:
            if r["topic"] == "Statins":
                continue
            detectable = statins_row["mean"] >= r["mde"]
            verdict = "WOULD have been detected" if detectable else "would NOT have been detected"
            md.append(
                f"- **{r['topic']}** (MDE = {r['mde']:.4f}): "
                f"a Statins-sized effect ({statins_row['mean']:+.4f}) "
                f"{verdict} at this topic's n and variance."
            )
        md.append("")
        md.append("### Reading")
        md.append("")
        md.append(
            "If the Statins-sized effect *would* have been detectable at "
            "Opioids/ADHD given their variance, the absence of a gap at "
            "those topics is informative — it argues against the pure "
            "statistical-power explanation. If the MDE is *larger* than "
            "the Statins effect, we cannot tell from this design alone "
            "whether the gap is absent or merely undetectable."
        )
        md.append("")
        md.append(
            "This analysis is approximate. The matched-n subsampling "
            "experiment in `parse_bow_experiments.py` gives the direct "
            "answer for Statins specifically; the power analysis above "
            "is the complementary cross-topic check."
        )

    out_path = OUTPUT_DIR / "power_analysis.md"
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")
    print("")
    print("Summary:")
    for r in rows:
        print(
            f"  {r['topic']:8s}  n_folds={r['n_folds']:3d}  "
            f"mean={r['mean']:+.4f}  MDE={r['mde']:.4f}"
        )


if __name__ == "__main__":
    main()
