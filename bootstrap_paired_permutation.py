"""Statistical analysis for the BERT × Cohen extension.

Loads per-fold WSS@95% values from the Colab .txt outputs and runs:

1. Bootstrap percentile CIs on the mean expert-vs-auto WSS@95% gap
   (per topic and pooled across topics)
2. Paired permutation test (exact enumeration when n_folds <= 20,
   else random permutations)
3. Nadeau-Bengio (2003) corrected resampled t-test for k-fold CV
   (more conservative variance estimator)

Inputs:
    Directory of .txt files following the pattern
    bert_{topic}_{mode}.txt produced by cohen_bert_pipeline.py.

Output:
    Console table + optional JSON dump of per-topic and pooled statistics.

H-Pub2 comparison being tested:
    expert mode (title_abstract_mesh) vs auto mode (auto_mesh)

The null is "no systematic per-fold difference in WSS@95% between the
two modes." Falsifying H-Pub2 requires showing the BoW expert advantage
(+0.121 on Statins) does NOT reproduce — i.e., the per-fold paired
differences are statistically indistinguishable from zero, or reversed.

Usage:
    python bootstrap_paired_permutation.py \\
        --input-dir colab_outputs \\
        --expert-mode title_abstract_mesh \\
        --auto-mode auto_mesh \\
        --output-json analysis_results.json
"""

import argparse
import json
import math
import re
import sys
from itertools import product
from pathlib import Path

import numpy as np


# Regex for the per-fold line in the .txt files. Handles negative WSS values.
# Example matches: "  Fold 1: acc=0.931 AUC=0.802 WSS@95=0.159 (train+pred 175s)"
#                  "  Fold 5: acc=0.975 AUC=0.745 WSS@95=-0.016 (train+pred 121s)"
FOLD_LINE_RE = re.compile(
    r"^\s*Fold\s+(\d+):\s+"
    r"acc=([\-\d\.]+)\s+"
    r"AUC=([\-\d\.]+)\s+"
    r"WSS@95=([\-\d\.]+)\s*\(",
    re.MULTILINE,
)


def parse_txt_file(path):
    """Extract per-fold metrics from a Colab .txt output.

    Returns dict with keys: fold_indices, accs, aucs, wsss (all lists).
    """
    text = Path(path).read_text(encoding="utf-8")
    matches = FOLD_LINE_RE.findall(text)
    if not matches:
        raise ValueError(f"No fold lines found in {path}")
    fold_indices = [int(m[0]) for m in matches]
    accs = [float(m[1]) for m in matches]
    aucs = [float(m[2]) for m in matches]
    wsss = [float(m[3]) for m in matches]
    return {
        "fold_indices": fold_indices,
        "accs": accs,
        "aucs": aucs,
        "wsss": wsss,
    }


def load_topic(input_dir, topic, modes):
    """Load per-fold WSS for all modes of one topic.

    Returns dict: mode -> list of per-fold WSS values.
    Missing modes are silently omitted; caller checks coverage.
    """
    out = {}
    for mode in modes:
        path = Path(input_dir) / f"bert_{topic.lower()}_{mode}.txt"
        if not path.exists():
            continue
        parsed = parse_txt_file(path)
        out[mode] = parsed["wsss"]
    return out


def paired_permutation_test(diffs, n_random=None, seed=42):
    """Two-sided paired permutation test on per-fold paired differences.

    Null: differences are symmetric around zero (equivalent to random
    sign assignment under exchangeability of paired observations).

    For n <= 20, enumerate all 2^n sign patterns exactly.
    For larger n, sample n_random permutations.

    Returns dict with p_value, observed_mean, and exactness flag.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    obs_mean = float(diffs.mean())

    if n <= 20:
        # Exact enumeration
        signs_iter = product([-1, 1], repeat=n)
        extreme = 0
        total = 0
        for signs in signs_iter:
            perm_mean = float(np.sum(np.asarray(signs) * diffs)) / n
            if abs(perm_mean) >= abs(obs_mean) - 1e-12:
                extreme += 1
            total += 1
        p = extreme / total
        return {"p_value": p, "observed_mean": obs_mean, "exact": True, "n_perms": total}

    # Monte Carlo
    if n_random is None:
        n_random = 20000
    rng = np.random.default_rng(seed)
    perm_signs = rng.choice([-1, 1], size=(n_random, n))
    perm_means = (perm_signs * diffs).mean(axis=1)
    # +1 in num+denom is the standard correction (Phipson & Smyth 2010)
    extreme = int(np.sum(np.abs(perm_means) >= abs(obs_mean) - 1e-12))
    p = (extreme + 1) / (n_random + 1)
    return {"p_value": p, "observed_mean": obs_mean, "exact": False, "n_perms": n_random}


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=42):
    """Percentile bootstrap CI on the mean of paired differences."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return {
        "mean": float(diffs.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "n_boot": n_boot,
        "alpha": alpha,
    }


def nadeau_bengio_t(diffs, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for k-fold CV.

    For k-fold CV with overlapping training sets, the naive paired
    t-test underestimates variance. The corrected statistic uses

        var_corrected = var_diff * (1/k + n_test / n_train)

    where var_diff is the sample variance of per-fold differences.
    Reference: Nadeau & Bengio (2003), Machine Learning 52(3):239-281.

    Returns t-stat, two-sided p-value, df, and the correction factor.
    """
    from scipy import stats

    diffs = np.asarray(diffs, dtype=float)
    k = len(diffs)
    mean_d = float(diffs.mean())
    # Sample variance (ddof=1)
    var_d = float(diffs.var(ddof=1)) if k > 1 else 0.0
    correction = 1.0 / k + n_test / n_train
    se_corrected = math.sqrt(var_d * correction) if var_d > 0 else 0.0
    if se_corrected == 0:
        return {
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "df": k - 1,
            "correction_factor": correction,
            "se_corrected": 0.0,
        }
    t_stat = mean_d / se_corrected
    df = k - 1
    p = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df)))
    return {
        "t_stat": float(t_stat),
        "p_value": p,
        "df": df,
        "correction_factor": correction,
        "se_corrected": se_corrected,
    }


# Cohen topic sizes (from the .txt headers Sam shared).
# n_train and n_test refer to per-fold sizes for 5-fold CV.
TOPIC_SIZES = {
    "statins": {"n_total": 2744, "n_test": 549, "n_train": 2195},
    "opiods":  {"n_total": 1772, "n_test": 354, "n_train": 1418},
    "adhd":    {"n_total": 803,  "n_test": 161, "n_train": 642},
}


def analyze_topic(topic, mode_to_wss, expert_mode, auto_mode):
    """Run all three tests on one topic's expert-vs-auto WSS comparison."""
    if expert_mode not in mode_to_wss or auto_mode not in mode_to_wss:
        return None
    expert = mode_to_wss[expert_mode]
    auto = mode_to_wss[auto_mode]
    if len(expert) != len(auto):
        raise ValueError(
            f"{topic}: expert ({len(expert)} folds) and auto ({len(auto)} folds) "
            "have different fold counts"
        )
    diffs = [e - a for e, a in zip(expert, auto)]

    sizes = TOPIC_SIZES.get(topic.lower(), {})
    result = {
        "topic": topic,
        "n_folds": len(diffs),
        "expert_mode": expert_mode,
        "auto_mode": auto_mode,
        "expert_wss_per_fold": expert,
        "auto_wss_per_fold": auto,
        "diffs_per_fold": diffs,
        "expert_mean": float(np.mean(expert)),
        "auto_mean": float(np.mean(auto)),
        "diff_mean": float(np.mean(diffs)),
        "bootstrap": bootstrap_ci(diffs),
        "permutation": paired_permutation_test(diffs),
    }
    if sizes:
        result["nadeau_bengio"] = nadeau_bengio_t(
            diffs, n_train=sizes["n_train"], n_test=sizes["n_test"]
        )
    return result


def analyze_pooled(per_topic_results):
    """Pool per-fold differences across topics for a higher-power test."""
    all_diffs = []
    for r in per_topic_results:
        all_diffs.extend(r["diffs_per_fold"])
    if not all_diffs:
        return None
    return {
        "n_folds_pooled": len(all_diffs),
        "topics_included": [r["topic"] for r in per_topic_results],
        "diff_mean": float(np.mean(all_diffs)),
        "bootstrap": bootstrap_ci(all_diffs),
        "permutation": paired_permutation_test(all_diffs),
    }


def format_report(per_topic, pooled, expert_mode, auto_mode):
    lines = []
    lines.append("=" * 78)
    lines.append(f"H-Pub2 test: {expert_mode} vs {auto_mode}")
    lines.append("Statistic: per-fold WSS@95% difference (expert − auto)")
    lines.append("=" * 78)

    header = (
        f"{'Topic':<10} {'k':>3} {'mean':>8} {'95% CI':>20} "
        f"{'perm p':>10} {'NB t':>8} {'NB p':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in per_topic:
        ci = r["bootstrap"]
        ci_str = f"[{ci['ci_lo']:+.3f}, {ci['ci_hi']:+.3f}]"
        perm_p = r["permutation"]["p_value"]
        nb = r.get("nadeau_bengio")
        nb_t = f"{nb['t_stat']:+.2f}" if nb else "  n/a"
        nb_p = f"{nb['p_value']:.4f}" if nb else "    n/a"
        lines.append(
            f"{r['topic']:<10} {r['n_folds']:>3} {r['diff_mean']:>+8.4f} "
            f"{ci_str:>20} {perm_p:>10.4f} {nb_t:>8} {nb_p:>10}"
        )
    if pooled:
        ci = pooled["bootstrap"]
        ci_str = f"[{ci['ci_lo']:+.3f}, {ci['ci_hi']:+.3f}]"
        lines.append("-" * len(header))
        lines.append(
            f"{'POOLED':<10} {pooled['n_folds_pooled']:>3} "
            f"{pooled['diff_mean']:>+8.4f} {ci_str:>20} "
            f"{pooled['permutation']['p_value']:>10.4f} {'  n/a':>8} {'    n/a':>10}"
        )
    lines.append("=" * 78)
    lines.append("")
    lines.append("Notes:")
    lines.append("  - mean: mean per-fold paired difference (expert − auto)")
    lines.append("  - 95% CI: bootstrap percentile interval (10k resamples)")
    lines.append("  - perm p: two-sided exact paired-permutation p-value")
    lines.append("            (5 folds = 32 sign patterns, min p = 0.0625;")
    lines.append("             15 pooled folds = 32,768 patterns)")
    lines.append("  - NB t/p: Nadeau-Bengio (2003) corrected resampled t-test")
    lines.append("            for k-fold CV. More conservative than naive paired t.")
    lines.append("  - Reference BoW expert-vs-auto gap on Statins: +0.121")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--expert-mode", type=str, default="title_abstract_mesh")
    parser.add_argument("--auto-mode", type=str, default="auto_mesh")
    parser.add_argument(
        "--topics", type=str, nargs="+",
        default=["Statins", "Opiods", "ADHD"],
        help="Cohen topics to analyze. Topics missing from input-dir are skipped.",
    )
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    all_modes = (
        "abstract", "title_abstract", "title_abstract_mesh", "auto_mesh",
        args.expert_mode, args.auto_mode,
    )

    per_topic_results = []
    coverage_notes = []
    for topic in args.topics:
        modes_data = load_topic(args.input_dir, topic, set(all_modes))
        if args.expert_mode not in modes_data or args.auto_mode not in modes_data:
            missing = [
                m for m in (args.expert_mode, args.auto_mode)
                if m not in modes_data
            ]
            coverage_notes.append(
                f"{topic}: missing {missing}, skipped"
            )
            continue
        r = analyze_topic(topic, modes_data, args.expert_mode, args.auto_mode)
        per_topic_results.append(r)

    if not per_topic_results:
        print("No topics with both expert and auto mode available.")
        for note in coverage_notes:
            print("  " + note)
        sys.exit(1)

    pooled = analyze_pooled(per_topic_results) if len(per_topic_results) > 1 else None

    print(format_report(per_topic_results, pooled, args.expert_mode, args.auto_mode))
    if coverage_notes:
        print("\nCoverage notes:")
        for note in coverage_notes:
            print("  " + note)

    if args.output_json:
        out = {
            "expert_mode": args.expert_mode,
            "auto_mode": args.auto_mode,
            "per_topic": per_topic_results,
            "pooled": pooled,
            "coverage_notes": coverage_notes,
        }
        Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
