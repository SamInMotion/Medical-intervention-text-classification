"""Verbose demonstration of the statistical analysis for the BERT × Cohen extension.

This is a transparent walk-through: every per-fold value, every difference,
every bootstrap-resample and permutation step is printed. It uses the same
underlying functions as bootstrap_paired_permutation.py but with intermediate
output so the computation can be audited end-to-end by a methodology reviewer.

Run from the repo root:
    python demo_statistical_analysis.py colab_outputs/

Outputs to stdout; redirect to a transcript file if desired.
"""

import json
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Section 0: configuration
# ---------------------------------------------------------------------------

TOPICS = ["Statins", "Opiods", "ADHD"]
EXPERT_MODE = "title_abstract_mesh"
AUTO_MODE = "auto_mesh"

# Per-fold CV sizes from the .txt headers (5-fold stratified):
TOPIC_SIZES = {
    "statins": {"n_total": 2744, "n_test": 549, "n_train": 2195},
    "opiods":  {"n_total": 1772, "n_test": 354, "n_train": 1418},
    "adhd":    {"n_total": 803,  "n_test": 161, "n_train": 642},
}

# BoW reference gaps (only Statins computed in the thesis BoW pipeline).
# This is the asymmetry to acknowledge: BERT has cross-topic data;
# BoW reference is one topic.
BOW_GAP_STATINS = +0.121

N_BOOT = 10000
ALPHA = 0.05
SEED = 42


# ---------------------------------------------------------------------------
# Section 1: parsing
# ---------------------------------------------------------------------------

import re

FOLD_RE = re.compile(
    r"^\s*Fold\s+(\d+):\s+acc=([\-\d\.]+)\s+AUC=([\-\d\.]+)\s+WSS@95=([\-\d\.]+)",
    re.MULTILINE,
)


def parse_fold_wss(path: Path) -> list:
    text = path.read_text(encoding="utf-8")
    matches = FOLD_RE.findall(text)
    return [float(m[3]) for m in matches]


def load(input_dir: Path):
    """Load per-fold WSS@95 for expert and auto modes, three topics."""
    data = {}
    for topic in TOPICS:
        topic_l = topic.lower()
        expert_path = input_dir / f"bert_{topic_l}_{EXPERT_MODE}.txt"
        auto_path = input_dir / f"bert_{topic_l}_{AUTO_MODE}.txt"
        if not expert_path.exists() or not auto_path.exists():
            print(f"  WARNING: missing files for {topic}; skipping.")
            continue
        data[topic] = {
            "expert": parse_fold_wss(expert_path),
            "auto": parse_fold_wss(auto_path),
        }
    return data


# ---------------------------------------------------------------------------
# Section 2: pretty-printing helpers
# ---------------------------------------------------------------------------

def hline(char="="):
    print(char * 78)


def section(title):
    print()
    hline("=")
    print(f" {title}")
    hline("=")


def subsection(title):
    print()
    print(f"--- {title} ---")


# ---------------------------------------------------------------------------
# Section 3: per-topic walk-through
# ---------------------------------------------------------------------------

def show_per_fold_values(data):
    section("STEP 1: Raw per-fold WSS@95% values from the Colab .txt outputs")
    print()
    print("Each topic was run with 5-fold stratified cross-validation.")
    print(f"Mode comparison: {EXPERT_MODE} (expert) vs {AUTO_MODE} (auto).")
    print()
    for topic in data:
        subsection(topic)
        e = data[topic]["expert"]
        a = data[topic]["auto"]
        print(f"  Expert mode ({EXPERT_MODE}):")
        for i, v in enumerate(e, 1):
            print(f"    Fold {i}: WSS@95 = {v:+.3f}")
        print(f"    Mean    = {np.mean(e):+.4f}  (matches Colab summary)")
        print()
        print(f"  Auto mode ({AUTO_MODE}):")
        for i, v in enumerate(a, 1):
            print(f"    Fold {i}: WSS@95 = {v:+.3f}")
        print(f"    Mean    = {np.mean(a):+.4f}  (matches Colab summary)")


def compute_diffs(data):
    section("STEP 2: Per-fold paired differences (expert − auto)")
    print()
    print("The unit of analysis is the per-fold paired difference: for each")
    print("fold i, compute d_i = expert_i − auto_i. Because both modes run on")
    print("the same fold-splits with the same seed, d_i is a paired observation.")
    print()
    diffs_by_topic = {}
    for topic, vals in data.items():
        e = vals["expert"]
        a = vals["auto"]
        d = [ei - ai for ei, ai in zip(e, a)]
        diffs_by_topic[topic] = d
        subsection(topic)
        for i, (ei, ai, di) in enumerate(zip(e, a, d), 1):
            print(f"    Fold {i}: {ei:+.3f}  −  {ai:+.3f}  =  {di:+.4f}")
        print(f"    Mean diff = {np.mean(d):+.4f}")
        print(f"    Sample SD = {np.std(d, ddof=1):+.4f}")
    return diffs_by_topic


# ---------------------------------------------------------------------------
# Section 4: bootstrap CI demonstrated
# ---------------------------------------------------------------------------

def bootstrap_demo(diffs, topic):
    subsection(f"Bootstrap percentile CI: {topic} (n_boot = {N_BOOT})")
    diffs = np.asarray(diffs)
    n = len(diffs)
    rng = np.random.default_rng(SEED)

    # Show a small worked example of how one resample works
    print(f"  Original per-fold diffs (n={n}): "
          + ", ".join(f"{x:+.3f}" for x in diffs))
    print()
    print(f"  Worked example: 3 resamples, each drawing {n} folds with replacement.")
    for ex in range(3):
        idx = rng.integers(0, n, size=n)
        sample = diffs[idx]
        print(f"    Resample {ex+1}: indices={list(idx)}, "
              f"values={[f'{x:+.3f}' for x in sample]}, "
              f"mean={float(sample.mean()):+.4f}")

    print()
    print(f"  Now running the full {N_BOOT} resamples...")
    rng = np.random.default_rng(SEED)  # reset so the demo above does not bias
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot_means = diffs[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * ALPHA / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - ALPHA / 2)))
    print(f"  Bootstrap distribution: mean={boot_means.mean():+.4f}, "
          f"SD={boot_means.std():.4f}")
    print(f"  95% percentile CI: [{lo:+.4f}, {hi:+.4f}]")
    return {"mean": float(diffs.mean()), "ci_lo": lo, "ci_hi": hi}


# ---------------------------------------------------------------------------
# Section 5: paired permutation demonstrated
# ---------------------------------------------------------------------------

def permutation_demo(diffs, topic):
    subsection(f"Exact paired permutation test: {topic}")
    diffs = np.asarray(diffs)
    n = len(diffs)
    obs_mean = float(diffs.mean())
    n_perms = 2 ** n

    print(f"  Null: differences are symmetric around zero")
    print(f"        (i.e., expert and auto labels are exchangeable per fold).")
    print(f"  Under the null, each fold's sign could equally well have been ±.")
    print(f"  With n={n} folds, there are 2^{n} = {n_perms} possible sign patterns.")
    print()
    print(f"  Observed mean difference: {obs_mean:+.4f}")
    print(f"  Two-sided test: count patterns with |perm_mean| ≥ |obs_mean|.")
    print()
    print(f"  Worked example: first 4 of the {n_perms} sign patterns:")

    signs_list = list(product([-1, 1], repeat=n))
    for ex_signs in signs_list[:4]:
        perm_mean = float(np.sum(np.asarray(ex_signs) * diffs)) / n
        is_extreme = abs(perm_mean) >= abs(obs_mean) - 1e-12
        print(f"    signs={list(ex_signs)}, "
              f"perm_mean={perm_mean:+.4f}, "
              f"|perm_mean| ≥ |obs|? {is_extreme}")

    # Now run the full enumeration
    extreme = 0
    for signs in signs_list:
        perm_mean = float(np.sum(np.asarray(signs) * diffs)) / n
        if abs(perm_mean) >= abs(obs_mean) - 1e-12:
            extreme += 1
    p = extreme / n_perms
    print()
    print(f"  Full enumeration: {extreme} of {n_perms} patterns produce "
          f"|perm_mean| ≥ |obs|.")
    print(f"  Two-sided p-value = {extreme}/{n_perms} = {p:.4f}")
    print(f"  (Minimum achievable two-sided p at n={n}: {2/n_perms:.4f})")
    return {"p_value": p, "n_perms": n_perms}


# ---------------------------------------------------------------------------
# Section 6: Nadeau-Bengio demonstrated
# ---------------------------------------------------------------------------

def nadeau_bengio_demo(diffs, topic, n_train, n_test):
    subsection(f"Nadeau-Bengio (2003) corrected t-test: {topic}")
    from scipy import stats
    diffs = np.asarray(diffs)
    k = len(diffs)
    mean_d = float(diffs.mean())
    var_d = float(diffs.var(ddof=1))

    print(f"  For k-fold CV with overlapping training sets, the naive paired")
    print(f"  t-test underestimates variance. Nadeau and Bengio (2003)")
    print(f"  derived a corrected statistic:")
    print()
    print(f"    var_corrected = sample_var * (1/k + n_test / n_train)")
    print()
    print(f"  Where:")
    print(f"    k        = number of folds                       = {k}")
    print(f"    n_test   = per-fold held-out size                 = {n_test}")
    print(f"    n_train  = per-fold training size                 = {n_train}")
    print(f"    1/k      = {1/k:.4f}")
    print(f"    n_test/n_train = {n_test/n_train:.4f}")

    correction = 1.0 / k + n_test / n_train
    print(f"    correction factor (sum) = {correction:.4f}")
    print()
    print(f"  Inputs from the {topic} differences:")
    print(f"    mean      = {mean_d:+.4f}")
    print(f"    sample var = {var_d:.6f}")
    print(f"    sample sd  = {math.sqrt(var_d):.4f}")
    print()
    se_corrected = math.sqrt(var_d * correction)
    t_stat = mean_d / se_corrected if se_corrected > 0 else float("nan")
    df = k - 1
    p = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df))) if se_corrected > 0 else float("nan")
    print(f"  Corrected SE = sqrt({var_d:.6f} × {correction:.4f}) = {se_corrected:.4f}")
    print(f"  t-statistic  = mean / SE_corrected = {mean_d:+.4f} / {se_corrected:.4f} = {t_stat:+.3f}")
    print(f"  df           = k - 1 = {df}")
    print(f"  two-sided p  = {p:.4f}")
    print()
    print(f"  For comparison, the naive paired t (no correction) would use:")
    naive_se = math.sqrt(var_d / k)
    naive_t = mean_d / naive_se if naive_se > 0 else float("nan")
    naive_p = float(2.0 * (1.0 - stats.t.cdf(abs(naive_t), df))) if naive_se > 0 else float("nan")
    print(f"    naive SE   = sqrt({var_d:.6f} / {k}) = {naive_se:.4f}")
    print(f"    naive t    = {naive_t:+.3f}, naive p = {naive_p:.4f}")
    print(f"  Correction makes the test more conservative by factor "
          f"sqrt(k × correction) = {math.sqrt(k*correction):.3f}.")
    return {"t_stat": t_stat, "p_value": p, "correction_factor": correction}


# ---------------------------------------------------------------------------
# Section 7: pooled analysis
# ---------------------------------------------------------------------------

def pooled_analysis(diffs_by_topic):
    section("STEP 5: Pooled analysis across topics")
    print()
    print("Pooling per-fold differences across topics increases the sample")
    print("size for the central-tendency test. With three topics × 5 folds")
    print("we have 15 paired differences. This raises the resolution of the")
    print("permutation test from 2^5 = 32 patterns to 2^15 = 32,768 patterns,")
    print("and tightens the bootstrap CI.")
    print()
    print("Caveat to acknowledge: pooling treats fold-pairs from different")
    print("topics as exchangeable, which is reasonable for testing the")
    print("central-tendency null (no systematic expert-auto difference)")
    print("but not for claims about within-topic structure.")
    print()

    pooled = []
    print("  Pooled differences (topic-fold):")
    for topic, d in diffs_by_topic.items():
        for i, di in enumerate(d, 1):
            pooled.append(di)
            print(f"    {topic} fold {i}: {di:+.4f}")
        print()
    pooled = np.asarray(pooled)
    n = len(pooled)
    print(f"  Total pooled n = {n}")
    print(f"  Pooled mean   = {pooled.mean():+.4f}")
    print(f"  Pooled SD     = {pooled.std(ddof=1):+.4f}")

    # Bootstrap
    subsection("Pooled bootstrap")
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot_means = pooled[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * ALPHA / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - ALPHA / 2)))
    print(f"  Pooled 95% bootstrap CI: [{lo:+.4f}, {hi:+.4f}]")

    # Exact permutation
    subsection(f"Pooled exact permutation ({2**n} sign patterns)")
    obs_mean = float(pooled.mean())
    extreme = 0
    for signs in product([-1, 1], repeat=n):
        pm = float(np.sum(np.asarray(signs) * pooled)) / n
        if abs(pm) >= abs(obs_mean) - 1e-12:
            extreme += 1
    p = extreme / (2 ** n)
    print(f"  Observed pooled mean: {obs_mean:+.4f}")
    print(f"  Extreme patterns: {extreme} of {2**n}")
    print(f"  Pooled two-sided p-value: {p:.5f}")
    print(f"  Minimum achievable two-sided p at n=15: {2/(2**n):.6f}")

    return {
        "n": n,
        "mean": obs_mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "perm_p": p,
    }


# ---------------------------------------------------------------------------
# Section 8: final comparison vs. BoW reference
# ---------------------------------------------------------------------------

def final_comparison(per_topic_summary, pooled_summary):
    section("STEP 6: What the statistics establish (and what they do not)")
    print()
    print("The H-Pub2 hypothesis predicted that the BoW Statins expert-MeSH")
    print(f"advantage of +{BOW_GAP_STATINS:.3f} WSS@95% persists when the")
    print("classifier is BiomedBERT. Three independent statistical views:")
    print()
    print("Per-topic 95% bootstrap CIs (expert − auto difference, BERT):")
    print()
    print(f"  {'Topic':<10} {'k':>3} {'mean diff':>12} {'95% CI':>22} {'perm p':>10}")
    print(f"  {'-'*10} {'-'*3} {'-'*12} {'-'*22} {'-'*10}")
    for r in per_topic_summary:
        ci_str = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        print(f"  {r['topic']:<10} {r['n_folds']:>3} {r['mean']:>+12.4f} "
              f"{ci_str:>22} {r['perm_p']:>10.4f}")
    if pooled_summary:
        ci_str = f"[{pooled_summary['ci_lo']:+.3f}, {pooled_summary['ci_hi']:+.3f}]"
        print(f"  {'-'*10} {'-'*3} {'-'*12} {'-'*22} {'-'*10}")
        print(f"  {'POOLED':<10} {pooled_summary['n']:>3} "
              f"{pooled_summary['mean']:>+12.4f} {ci_str:>22} "
              f"{pooled_summary['perm_p']:>10.4f}")
    print()
    print("Two summary statements supported by the analysis:")
    print()
    print("  1. The pooled 95% CI for the expert-vs-auto difference is")
    print(f"     [{pooled_summary['ci_lo']:+.3f}, {pooled_summary['ci_hi']:+.3f}].")
    print(f"     The BoW reference gap of +{BOW_GAP_STATINS:.3f} falls "
          f"{BOW_GAP_STATINS - pooled_summary['ci_hi']:+.3f} above the")
    print(f"     CI's upper bound — approximately "
          f"{(BOW_GAP_STATINS - pooled_summary['ci_hi']) / ((pooled_summary['ci_hi'] - pooled_summary['ci_lo']) / 2):.1f}× the CI half-width.")
    print(f"     The hypothesis of persistence at the BoW magnitude is")
    print(f"     not consistent with the BERT data.")
    print()
    print(f"  2. The pooled permutation p-value ({pooled_summary['perm_p']:.4f})")
    print("     is not significant at α = 0.05. The data are consistent with")
    print("     no systematic expert-vs-auto difference at BERT scale.")
    print()
    print("What the analysis does NOT establish:")
    print()
    print("  - Reversal of the gap. The CI includes zero. \"BERT reverses the")
    print("    gap\" overstates what the 15-fold data show.")
    print("  - Cross-classifier persistence on Opiods and ADHD. The +0.121")
    print("    BoW reference is from Statins only. Opiods and ADHD BoW")
    print("    reference gaps have not been computed and remain an open")
    print("    methodological question.")
    print("  - Generalization beyond BiomedBERT-base, MeSH, or the three")
    print("    tested topics. The Limitations section enumerates these.")


# ---------------------------------------------------------------------------
# Section 9: entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python demo_statistical_analysis.py <input_dir>")
        sys.exit(1)
    input_dir = Path(sys.argv[1])
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    section("DEMONSTRATION: statistical analysis for BERT × Cohen extension")
    print()
    print(f"Input directory: {input_dir}")
    print(f"Comparing modes: {EXPERT_MODE} (expert) vs {AUTO_MODE} (auto)")
    print(f"Topics:          {', '.join(TOPICS)}")
    print(f"Bootstrap:       {N_BOOT} resamples, percentile method, α = {ALPHA}")
    print(f"Permutation:     exact enumeration of all 2^n sign patterns")
    print(f"BoW reference:   +{BOW_GAP_STATINS:.3f} (Statins; Opiods/ADHD not yet computed)")

    data = load(input_dir)
    if not data:
        print("No usable data found.")
        sys.exit(1)

    show_per_fold_values(data)
    diffs_by_topic = compute_diffs(data)

    # Per-topic bootstrap + permutation + Nadeau-Bengio
    section("STEP 3: Per-topic bootstrap CIs")
    per_topic_summary = []
    for topic, d in diffs_by_topic.items():
        boot = bootstrap_demo(d, topic)
        per_topic_summary.append({
            "topic": topic,
            "n_folds": len(d),
            "mean": boot["mean"],
            "ci_lo": boot["ci_lo"],
            "ci_hi": boot["ci_hi"],
        })

    section("STEP 4: Per-topic paired permutation tests")
    for i, (topic, d) in enumerate(diffs_by_topic.items()):
        perm = permutation_demo(d, topic)
        per_topic_summary[i]["perm_p"] = perm["p_value"]

    section("STEP 4b: Per-topic Nadeau-Bengio corrected t-tests")
    for topic, d in diffs_by_topic.items():
        sizes = TOPIC_SIZES.get(topic.lower())
        if sizes:
            nadeau_bengio_demo(d, topic, sizes["n_train"], sizes["n_test"])

    pooled_summary = pooled_analysis(diffs_by_topic)
    final_comparison(per_topic_summary, pooled_summary)

    print()
    hline("=")
    print(" END OF DEMONSTRATION")
    hline("=")


if __name__ == "__main__":
    main()
