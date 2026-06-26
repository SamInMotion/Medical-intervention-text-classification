"""Parse BoW multi-run outputs and characterize the expert-auto MeSH gap distribution.

Minimal extension of the original Statins-only parser (commit history: this file)
to support Opiods and ADHD as well. Same regex, same JSON output structure, same
per-fold tracking. The only changes are:
  - The hardcoded "Statins" string in the section header regex becomes a parameter.
  - The hardcoded input/output file paths become topic-derived.
  - A --topic argument selects the topic.
  - A --all flag runs Statins + Opiods + ADHD in sequence.

Usage from repo root:
    python scripts/parse_bow_multirun.py                      # Statins (default, back-compat)
    python scripts/parse_bow_multirun.py --topic Opiods
    python scripts/parse_bow_multirun.py --topic ADHD
    python scripts/parse_bow_multirun.py --all                # all three topics

Output: outputs/bow_<topic_lower>_multirun_summary.json with the same schema as
the existing Statins summary, so downstream code (make_paper_artifacts.py,
make_fig1_v2.py, paper draft) reads it without modification.
"""
import argparse
import json
import re
from pathlib import Path
from statistics import mean, stdev

MODES = ["abstract", "title_abstract", "title_abstract_mesh", "auto_mesh"]
N_RUNS = 7


def parse_run(path, topic):
    """Parse one bow_<topic>_run<N>.txt produced by `--compare-text-modes`.

    Returns dict keyed by mode, value is list of 5 Reg-classifier WSS@95 values.
    """
    text = Path(path).read_text()
    result = {}
    for mode in MODES:
        # Section header in the pipeline output is:
        #     Cohen topic: <Topic> (<mode> mode)
        # The original Statins parser anchored on "Statins \(<mode> mode\)";
        # we generalise by parameterising the topic. The mode name
        # disambiguates title_abstract vs title_abstract_mesh because
        # re.escape() handles the underscore correctly and the trailing
        # " mode)" anchors the match.
        pattern = rf"{re.escape(topic)} \({re.escape(mode)} mode\)(.*?)(?=Cohen topic:|TEXT MODE COMPARISON|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            result[mode] = [None] * 5
            continue
        section = match.group(1)
        # Per-fold pattern: every Fold line has both BL and Reg WSS values;
        # the lazy ".*?" between "Fold N:" and "Reg" then ".*?" between
        # "Reg" and "WSS=" together pick the Reg WSS, not the BL WSS.
        fold_matches = re.findall(r"Fold (\d+):.*?Reg .*?WSS=(-?\d+\.\d+)", section)
        folds = sorted(fold_matches, key=lambda x: int(x[0]))[:5]
        result[mode] = [float(w) for _, w in folds]
    return result


def summarise_topic(topic, runs_dir, output_dir, n_runs=N_RUNS):
    topic_lower = topic.lower()
    files = [runs_dir / f"bow_{topic_lower}_run{i}.txt" for i in range(1, n_runs + 1)]

    runs = []
    for f in files:
        if f.exists():
            runs.append((str(f), parse_run(f, topic)))
        else:
            print(f"SKIP: {f} not found")

    print(f"\nParsed {len(runs)} runs for {topic}.\n")

    print(f"{'Run':<28} {'Expert':>8} {'Auto':>8} {'Gap':>9}   {'Per-fold gap':<35}")
    print("-" * 95)

    gap_per_run = []
    gaps_per_fold_all = []
    for path, modes in runs:
        expert = modes["title_abstract_mesh"]
        auto = modes["auto_mesh"]
        if None in expert or None in auto:
            print(f"{Path(path).stem}: parse failed, skipping")
            continue
        per_fold = [e - a for e, a in zip(expert, auto)]
        gap = mean(per_fold)
        gap_per_run.append(gap)
        gaps_per_fold_all.extend(per_fold)
        fold_str = " ".join(f"{g:+.3f}" for g in per_fold)
        print(f"{Path(path).stem:<28} {mean(expert):>8.4f} {mean(auto):>8.4f} {gap:>+9.4f}   {fold_str}")

    if not gap_per_run:
        print(f"\nNo parsable runs for {topic}. Skipping summary.")
        return None

    print("\n" + "=" * 60)
    print(f"BoW {topic} expert - auto MeSH gap, multi-run summary")
    print("=" * 60)
    print(f"Number of runs:          {len(gap_per_run)}")
    print(f"Per-run gap mean:        {mean(gap_per_run):+.4f}")
    if len(gap_per_run) > 1:
        print(f"Per-run gap stdev:       {stdev(gap_per_run):.4f}")
    print(f"Per-run gap range:       [{min(gap_per_run):+.4f}, {max(gap_per_run):+.4f}]")
    print(f"Total folds observed:    {len(gaps_per_fold_all)}")
    print(f"Folds strictly positive: {sum(1 for g in gaps_per_fold_all if g > 0)}/{len(gaps_per_fold_all)}")
    print(f"Folds exactly zero:      {sum(1 for g in gaps_per_fold_all if g == 0)}/{len(gaps_per_fold_all)}")
    print(f"Folds negative:          {sum(1 for g in gaps_per_fold_all if g < 0)}/{len(gaps_per_fold_all)}")

    # JSON output: identical schema to the original Statins summary so
    # downstream code (make_paper_artifacts.py, make_fig1_v2.py) consumes
    # it without modification.
    out = {
        "runs": [
            {"file": Path(p).stem, "modes": m} for p, m in runs
        ],
        "summary": {
            "n_runs": len(gap_per_run),
            "gap_mean": mean(gap_per_run),
            "gap_stdev": stdev(gap_per_run) if len(gap_per_run) > 1 else None,
            "gap_min": min(gap_per_run),
            "gap_max": max(gap_per_run),
            "n_folds_total": len(gaps_per_fold_all),
            "n_folds_positive": sum(1 for g in gaps_per_fold_all if g > 0),
            "n_folds_zero": sum(1 for g in gaps_per_fold_all if g == 0),
            "n_folds_negative": sum(1 for g in gaps_per_fold_all if g < 0),
        },
    }
    out_path = output_dir / f"bow_{topic_lower}_multirun_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--topic", type=str, default="Statins",
                        help="Topic name matching the cache spelling and run filenames. "
                             "Default: Statins (back-compat with existing usage).")
    parser.add_argument("--all", action="store_true",
                        help="Run for Statins, Opiods, and ADHD in sequence")
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs"),
                        help="Directory containing bow_<topic>_run<N>.txt files (default: outputs/)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Output directory for summary JSON (default: outputs/)")
    parser.add_argument("--n-runs", type=int, default=N_RUNS,
                        help=f"Number of runs to look for (default: {N_RUNS})")
    args = parser.parse_args()

    topics = ["Statins", "Opiods", "ADHD"] if args.all else [args.topic]
    for topic in topics:
        summarise_topic(topic, args.runs_dir, args.output_dir, n_runs=args.n_runs)
        print()


if __name__ == "__main__":
    main()
