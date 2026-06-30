"""Parse and analyse outputs from the BoW subsampling and 10-fold experiments.

Reads:
  paper_experiments/outputs/bow_statins_subN803_subseed{1..7}_modes.txt
  paper_experiments/outputs/bow_statins_kfold10_run{1..7}_modes.txt

Computes per-fold expert-vs-auto WSS@95 gaps, pools across reruns, runs a
10,000-resample percentile bootstrap CI on each condition. Compares against
the existing Statins full-n and ADHD reference values pulled from
bow_stats_results.json.

Output:
  paper_experiments/outputs/bow_experiments_summary.csv      (long format, per-fold)
  paper_experiments/outputs/bow_experiments_summary.md       (comparison table)
  paper_experiments/outputs/bow_experiments_decision.txt     (verdict for §5.2)

Usage:
    python paper_experiments/parse_bow_experiments.py
"""

import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path("paper_experiments/outputs")
REPO_ROOT = Path(".")
EXPERT_MODE = "title_abstract_mesh"
AUTO_MODE = "auto_mesh"

# Regex matches the per-fold line emitted by cohen_pipeline.py:
#   Fold 1: BL acc=0.945 AUC=0.812 WSS=0.123  Reg acc=0.945 AUC=0.823 WSS=0.234
FOLD_RE = re.compile(
    r"Fold\s+(\d+):\s+BL\s+acc=([\-\d.]+)\s+AUC=([\-\d.]+)\s+WSS=([\-\d.]+)\s+"
    r"Reg\s+acc=([\-\d.]+)\s+AUC=([\-\d.]+)\s+WSS=([\-\d.]+)",
    re.IGNORECASE,
)

# Block header from cohen_pipeline.py looks like:
#   Cohen topic: Statins (title_abstract_mesh mode)
BLOCK_RE = re.compile(r"Cohen topic:\s+(\S+)\s+\(([^)]+)\s+mode\)")

# Sample count line:
#   5-fold CV on 803 abstracts (44 included, 759 excluded)
SAMPLES_RE = re.compile(r"(\d+)-fold CV on (\d+) abstracts \((\d+) included")

N_BOOT = 10_000
ALPHA = 0.05
SEED = 42


def parse_modes_file(path):
    """Return dict[mode] = {'n_total': int, 'n_included': int, 'wss': [float, ...]}.

    Reads a --compare-text-modes output file. Each text mode block contains
    the same `Cohen topic: X (MODE mode)` header followed by the per-fold lines.
    """
    text = path.read_text(encoding="utf-8")
    by_mode = {}
    blocks = re.split(r"\nCohen topic:", text)
    if len(blocks) > 1:
        blocks = ["Cohen topic:" + b for b in blocks[1:]]
    for block in blocks:
        m_block = BLOCK_RE.search(block)
        if not m_block:
            continue
        mode = m_block.group(2).strip()
        m_samp = SAMPLES_RE.search(block)
        n_total = int(m_samp.group(2)) if m_samp else None
        n_inc = int(m_samp.group(3)) if m_samp else None
        wss_values = []
        for m in FOLD_RE.finditer(block):
            wss_values.append(float(m.group(7)))  # Reg WSS
        if wss_values:
            by_mode[mode] = {
                "n_total": n_total,
                "n_included": n_inc,
                "wss": wss_values,
            }
    return by_mode


def collect_experiment(pattern, run_id_re):
    """Walk OUTPUT_DIR for files matching pattern, parse each, return long list."""
    rows = []
    files = sorted(OUTPUT_DIR.glob(pattern))
    if not files:
        return rows, []
    parsed_runs = []
    for fp in files:
        m = run_id_re.search(fp.name)
        if not m:
            continue
        run_id = m.group(1)
        by_mode = parse_modes_file(fp)
        if EXPERT_MODE not in by_mode or AUTO_MODE not in by_mode:
            print(f"[warn] {fp.name}: missing expected modes, skipping", file=sys.stderr)
            continue
        n_total = by_mode[EXPERT_MODE]["n_total"]
        for fold_idx, (e, a) in enumerate(
            zip(by_mode[EXPERT_MODE]["wss"], by_mode[AUTO_MODE]["wss"]), start=1
        ):
            rows.append({
                "run_id": run_id,
                "n_total": n_total,
                "fold": fold_idx,
                "expert_wss": e,
                "auto_wss": a,
                "diff": e - a,
            })
        parsed_runs.append(fp.name)
    return rows, parsed_runs


def bootstrap_ci(diffs, n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diffs[idx].mean(axis=1)
    return {
        "mean": float(diffs.mean()),
        "ci_lo": float(np.percentile(means, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "n": n,
        "sd": float(diffs.std(ddof=1)) if n > 1 else float("nan"),
    }


def write_long_csv(all_rows, out_path):
    fieldnames = ["condition", "run_id", "n_total", "fold", "expert_wss", "auto_wss", "diff"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)


def load_reference_bow_stats():
    """Try to pull existing Statins-full and ADHD reference distributions.

    Looks for bow_stats_results.json in two likely locations. Returns
    None if not found -- the analysis will still run, just without the
    reference comparison.
    """
    for candidate in [
        REPO_ROOT / "bow_stats_results.json",
        REPO_ROOT / "paper_experiments" / "bow_stats_results.json",
        Path("/g/My Drive/cohen_bert_run/bow_stats_results.json"),
    ]:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


def reference_diffs_from_stats(stats, topic):
    """Pull per-fold diffs for a topic from bow_stats_results.json.

    Format may vary across consolidations. This function tries common shapes
    and returns either a list of per-fold diffs or None if it cannot find them.
    """
    if not stats:
        return None
    candidates = []
    if isinstance(stats, dict):
        if "per_topic" in stats and isinstance(stats["per_topic"], list):
            for entry in stats["per_topic"]:
                if entry.get("topic") == topic and "diffs" in entry:
                    candidates.append(entry["diffs"])
        if topic in stats and isinstance(stats[topic], dict):
            entry = stats[topic]
            if "diffs" in entry:
                candidates.append(entry["diffs"])
            if "per_run" in entry and isinstance(entry["per_run"], list):
                pooled = []
                for r in entry["per_run"]:
                    if "diffs" in r:
                        pooled.extend(r["diffs"])
                if pooled:
                    candidates.append(pooled)
    return candidates[0] if candidates else None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Experiment A: subsampling
    sub_rows, sub_files = collect_experiment(
        "bow_statins_subN*_subseed*_modes.txt",
        re.compile(r"subseed(\d+)"),
    )

    # Experiment B: 10-fold
    tenfold_rows, tenfold_files = collect_experiment(
        "bow_statins_kfold10_run*_modes.txt",
        re.compile(r"run(\d+)"),
    )

    if not sub_rows and not tenfold_rows:
        print(
            "No experiment outputs found yet under "
            f"{OUTPUT_DIR}/. Run the *.sh scripts first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Tag each row with its condition
    for r in sub_rows:
        r["condition"] = "statins_sub_n803_kfold5"
    for r in tenfold_rows:
        r["condition"] = "statins_full_kfold10"

    all_rows = sub_rows + tenfold_rows
    write_long_csv(all_rows, OUTPUT_DIR / "bow_experiments_summary.csv")

    # Compute pooled CIs per condition
    summaries = {}
    if sub_rows:
        summaries["statins_sub_n803_kfold5"] = bootstrap_ci([r["diff"] for r in sub_rows])
        summaries["statins_sub_n803_kfold5"]["n_runs"] = len({r["run_id"] for r in sub_rows})
    if tenfold_rows:
        summaries["statins_full_kfold10"] = bootstrap_ci([r["diff"] for r in tenfold_rows])
        summaries["statins_full_kfold10"]["n_runs"] = len({r["run_id"] for r in tenfold_rows})

    # Reference values
    stats = load_reference_bow_stats()
    for ref_topic, label in [("Statins", "statins_full_kfold5"), ("ADHD", "adhd_full_kfold5"),
                              ("Opiods", "opioids_full_kfold5")]:
        diffs = reference_diffs_from_stats(stats, ref_topic)
        if diffs:
            summaries[label] = bootstrap_ci(diffs)
            summaries[label]["n_runs"] = None

    # Write markdown comparison table
    md_lines = [
        "# BoW Experiments Summary",
        "",
        "Pooled per-fold expert-vs-auto WSS@95 gap, 95% percentile bootstrap CIs"
        " (10,000 resamples).",
        "",
        "| Condition | n folds | n runs | Mean gap | 95% CI | SD |",
        "|---|---|---|---|---|---|",
    ]
    nice_label = {
        "statins_full_kfold5":   "Statins full (n≈2,744), 5-fold (reference)",
        "statins_sub_n803_kfold5": "Statins subsampled (n=803), 5-fold (NEW)",
        "statins_full_kfold10":  "Statins full (n≈2,744), 10-fold (NEW)",
        "adhd_full_kfold5":      "ADHD full (n≈803), 5-fold (reference)",
        "opioids_full_kfold5":   "Opioids full (n≈1,772), 5-fold (reference)",
    }
    order = [
        "statins_full_kfold5",
        "statins_sub_n803_kfold5",
        "statins_full_kfold10",
        "adhd_full_kfold5",
        "opioids_full_kfold5",
    ]
    for k in order:
        if k not in summaries:
            continue
        s = summaries[k]
        sd_str = f"{s.get('sd', float('nan')):.4f}" if not math.isnan(s.get("sd", float("nan"))) else "—"
        n_runs_str = str(s.get("n_runs")) if s.get("n_runs") else "—"
        md_lines.append(
            f"| {nice_label.get(k, k)} | {s['n']} | {n_runs_str} | "
            f"{s['mean']:+.4f} | [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] | {sd_str} |"
        )
    md_lines.append("")
    md_lines.append("## Files parsed")
    md_lines.append("")
    md_lines.append("Subsampling experiment:")
    for fn in sub_files:
        md_lines.append(f"  - {fn}")
    md_lines.append("")
    md_lines.append("10-fold experiment:")
    for fn in tenfold_files:
        md_lines.append(f"  - {fn}")

    (OUTPUT_DIR / "bow_experiments_summary.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    # Decision rule for §5.2
    verdict_lines = ["# Decision verdict for §5.2", ""]
    if "statins_sub_n803_kfold5" in summaries and "statins_full_kfold5" in summaries:
        sub = summaries["statins_sub_n803_kfold5"]
        full = summaries["statins_full_kfold5"]
        adhd = summaries.get("adhd_full_kfold5")
        verdict_lines.append(f"Statins full mean gap:       {full['mean']:+.4f} (CI {full['ci_lo']:+.4f}, {full['ci_hi']:+.4f})")
        verdict_lines.append(f"Statins subsampled n=803:    {sub['mean']:+.4f} (CI {sub['ci_lo']:+.4f}, {sub['ci_hi']:+.4f})")
        if adhd:
            verdict_lines.append(f"ADHD full n≈803 reference:   {adhd['mean']:+.4f} (CI {adhd['ci_lo']:+.4f}, {adhd['ci_hi']:+.4f})")
        verdict_lines.append("")
        threshold = 0.04  # roughly half the full Statins effect; below this is too weak to claim linguistic structure as cause
        if sub["ci_lo"] > 0 and sub["mean"] > threshold:
            verdict_lines.append(
                f"VERDICT: Statins gap persists at matched n=803 (CI lower bound {sub['ci_lo']:+.4f} > 0,"
                f" mean {sub['mean']:+.4f} > threshold {threshold}). The semasiological/onomasiological"
                " argument in §5.2 is empirically defended against the data-confound interpretation."
            )
            verdict_lines.append("")
            verdict_lines.append("Next: run the BERT Statins subsampling confirmation on Colab T4.")
        elif sub["ci_lo"] > 0 and sub["mean"] <= threshold:
            verdict_lines.append(
                f"VERDICT: Statins gap persists at matched n=803 but the magnitude (mean {sub['mean']:+.4f})"
                f" is below the linguistic-structure threshold ({threshold}). §5.2 needs a careful"
                " hedge: the effect survives matched-n at reduced magnitude, which is consistent"
                " with both linguistic-structure and partial data-confound explanations."
            )
        else:
            verdict_lines.append(
                f"VERDICT: Statins gap does NOT robustly persist at matched n=803 (CI includes zero:"
                f" [{sub['ci_lo']:+.4f}, {sub['ci_hi']:+.4f}]). The §5.2 semasiological/onomasiological"
                " argument needs reframing. The topic-stratified pattern is consistent with statistical"
                " power at smaller corpus sizes rather than linguistic structure."
            )
            verdict_lines.append("")
            verdict_lines.append(
                "Next: do NOT run the BERT subsampling confirmation. Rework §5.2 to acknowledge"
                " the power explanation. The paper is still publishable but with a different framing."
            )

    if "statins_full_kfold10" in summaries and "statins_full_kfold5" in summaries:
        ten = summaries["statins_full_kfold10"]
        five = summaries["statins_full_kfold5"]
        verdict_lines.append("")
        verdict_lines.append(f"10-fold sensitivity check: {ten['mean']:+.4f} (CI {ten['ci_lo']:+.4f}, {ten['ci_hi']:+.4f}) "
                              f"vs 5-fold reference {five['mean']:+.4f}.")
        if abs(ten["mean"] - five["mean"]) < 0.02:
            verdict_lines.append("Pattern replicates at 10-fold. Christer Q4 answered with data.")
        else:
            verdict_lines.append("Notable shift between 5-fold and 10-fold. Investigate before paper revision.")

    (OUTPUT_DIR / "bow_experiments_decision.txt").write_text(
        "\n".join(verdict_lines) + "\n", encoding="utf-8"
    )

    print("Wrote:")
    print(f"  {OUTPUT_DIR / 'bow_experiments_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'bow_experiments_summary.md'}")
    print(f"  {OUTPUT_DIR / 'bow_experiments_decision.txt'}")
    print("")
    print("Read bow_experiments_decision.txt for the §5.2 verdict.")


if __name__ == "__main__":
    main()
