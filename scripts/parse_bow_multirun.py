"""Parse 7 BoW Statins runs and characterize the expert-auto MeSH gap distribution."""
import re
from pathlib import Path
from statistics import mean, stdev

FILES = [f"outputs/bow_statins_run{i}.txt" for i in range(1, 8)]
MODES = ["abstract", "title_abstract", "title_abstract_mesh", "auto_mesh"]


def parse_run(path):
    text = Path(path).read_text()
    result = {}
    for mode in MODES:
        # Use a delimiter that doesn't confuse title_abstract vs title_abstract_mesh
        pattern = rf"Statins \({re.escape(mode)} mode\)(.*?)(?=Cohen topic:|TEXT MODE COMPARISON|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            result[mode] = [None] * 5
            continue
        section = match.group(1)
        fold_matches = re.findall(r"Fold (\d+):.*?Reg .*?WSS=(-?\d+\.\d+)", section)
        folds = sorted(fold_matches, key=lambda x: int(x[0]))[:5]
        result[mode] = [float(w) for _, w in folds]
    return result


runs = []
for f in FILES:
    if Path(f).exists():
        runs.append((f, parse_run(f)))
    else:
        print(f"SKIP: {f} not found")

print(f"\nParsed {len(runs)} runs.\n")

print(f"{'Run':<28} {'Expert':>8} {'Auto':>8} {'Gap':>9}   {'Per-fold gap':<35}")
print("-" * 95)

gap_per_run = []
gaps_per_fold_all = []
for path, modes in runs:
    expert, auto = modes["title_abstract_mesh"], modes["auto_mesh"]
    if None in expert or None in auto:
        print(f"{Path(path).stem}: parse failed, skipping")
        continue
    per_fold = [e - a for e, a in zip(expert, auto)]
    gap = mean(per_fold)
    gap_per_run.append(gap)
    gaps_per_fold_all.extend(per_fold)
    fold_str = " ".join(f"{g:+.3f}" for g in per_fold)
    print(f"{Path(path).stem:<28} {mean(expert):>8.4f} {mean(auto):>8.4f} {gap:>+9.4f}   {fold_str}")

print("\n" + "=" * 60)
print("BoW Statins expert - auto MeSH gap, multi-run summary")
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

# Save consolidated JSON for later use in paper artifacts
import json
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
Path("outputs/bow_statins_multirun_summary.json").write_text(json.dumps(out, indent=2))
print("\nSaved: outputs/bow_statins_multirun_summary.json")
