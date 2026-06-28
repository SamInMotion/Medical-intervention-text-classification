"""Per-fold paired bootstrap for BERT multi-seed Opioids/Statins/ADHD.

Loads the three multi-seed summary JSONs produced by the Colab notebook
(Section 6 cell), extracts per-fold WSS@95% values for the expert MeSH and
auto MeSH modes, computes per-fold paired differences, and reports per-topic
and pooled-three-topic bootstrap confidence intervals plus exact (or sampled)
permutation p-values.

The results are paper-ready: drop the printed table into Section 4.3.

Usage from repo root:
    python scripts/bootstrap_bert_per_fold.py

Inputs (in outputs/):
    bert_statins_multiseed_summary.json
    bert_opiods_multiseed_summary.json
    bert_adhd_multiseed_summary.json

Outputs:
    Console table ready for paper Table 5.
    outputs/bert_per_fold_bootstrap.json    archive of all numbers.
"""

import json
from itertools import product
from pathlib import Path

import numpy as np


TOPICS = ["Statins", "Opiods", "ADHD"]
TOPIC_DISPLAY = {"Statins": "Statins", "Opiods": "Opioids", "ADHD": "ADHD"}
OUTPUTS_DIR = Path("outputs")
N_BOOT = 10000
N_PERM_SAMPLED = 100000
SEED = 42


def load_per_fold_diffs(topic):
    """Returns list of per-fold expert-auto WSS@95 paired diffs."""
    path = OUTPUTS_DIR / f"bert_{topic.lower()}_multiseed_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expert_runs = payload["expert_runs"]
    auto_runs = payload["auto_runs"]

    diffs = []
    for seed in expert_runs:
        if seed not in auto_runs:
            print(f"  WARN: seed {seed} missing from auto_runs for {topic}")
            continue
        e_folds = expert_runs[seed]["folds"]
        a_folds = auto_runs[seed]["folds"]
        if len(e_folds) != len(a_folds):
            print(f"  WARN: fold count mismatch for {topic} seed {seed}")
            continue
        for e_fold, a_fold in zip(e_folds, a_folds):
            e_wss = e_fold.get("wss_at_95")
            a_wss = a_fold.get("wss_at_95")
            if e_wss is not None and a_wss is not None:
                diffs.append(e_wss - a_wss)
    return diffs


def boot_ci(values, n_boot=N_BOOT, alpha=0.05, seed=SEED):
    """Percentile bootstrap CI on the mean."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def perm_p(values, seed=SEED):
    """Two-sided sign-flip paired permutation p-value.

    Exact enumeration for n <= 20; random sampling otherwise.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    observed = abs(arr.mean())
    if n <= 20:
        count = 0
        total = 0
        for signs in product([-1, 1], repeat=n):
            if abs(np.dot(signs, arr) / n) >= observed:
                count += 1
            total += 1
        return count / total
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(N_PERM_SAMPLED, n))
    stats = np.abs((signs * arr).mean(axis=1))
    return float(((stats >= observed).sum() + 1) / (N_PERM_SAMPLED + 1))


def nadeau_bengio_t(values, k=5):
    """Nadeau-Bengio corrected paired t-test for k-fold CV.

    Correction factor: 1/k + n_test/n_train = 1/k + 1/(k-1).
    Returns (t, p_two_sided) using the standard t distribution with df=n-1.

    Note: this is a coarse approximation that treats all n observations as if
    they came from a single k-fold CV. The multi-seed data has n=k*n_seeds
    observations across seeds; we apply the correction at the fold level only.
    Caveat documented in the printed output.
    """
    from scipy import stats as scipy_stats
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < 2:
        return float("nan"), float("nan")
    mean = arr.mean()
    var = arr.var(ddof=1)
    correction = 1.0 / k + 1.0 / (k - 1)
    se = (var * correction) ** 0.5
    if se == 0:
        return float("nan"), float("nan")
    t = mean / (se / (n ** 0.5))
    p = 2 * (1 - scipy_stats.t.cdf(abs(t), df=n - 1))
    return float(t), float(p)


def main():
    diffs_by_topic = {}
    print("Loading multi-seed summaries...")
    for topic in TOPICS:
        diffs = load_per_fold_diffs(topic)
        diffs_by_topic[topic] = diffs
        print(f"  {TOPIC_DISPLAY[topic]:<10} loaded n={len(diffs)} per-fold paired diffs")

    pooled = []
    for topic in TOPICS:
        pooled.extend(diffs_by_topic[topic])

    print()
    print("=" * 96)
    print("BERT multi-seed per-fold paired bootstrap (BERT side of paper Table 5)")
    print("=" * 96)
    print(f"{'Classifier':<18} {'Topic':<10} {'n':>4} {'Mean':>9} {'SD':>8} {'95% CI':>22} {'Perm p':>8}")
    print("-" * 96)

    results = {"per_fold": {}}

    for topic in TOPICS:
        diffs = diffs_by_topic[topic]
        if not diffs:
            print(f"{'BERT (multi-seed)':<18} {TOPIC_DISPLAY[topic]:<10} n=0 skipped")
            continue
        arr = np.asarray(diffs)
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        lo, hi = boot_ci(diffs)
        p = perm_p(diffs)
        ci_str = f"[{lo:+.3f}, {hi:+.3f}]"
        print(f"{'BERT (multi-seed)':<18} {TOPIC_DISPLAY[topic]:<10} {len(arr):>4} {mean:>+9.4f} {sd:>8.4f} {ci_str:>22} {p:>8.4f}")
        results["per_fold"][topic] = {
            "n": len(arr), "mean": mean, "sd": sd, "lo": lo, "hi": hi, "perm_p": p,
        }

    # Pooled
    arr = np.asarray(pooled)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    lo, hi = boot_ci(pooled)
    p = perm_p(pooled)
    ci_str = f"[{lo:+.3f}, {hi:+.3f}]"
    print(f"{'BERT (multi-seed)':<18} {'POOLED':<10} {len(arr):>4} {mean:>+9.4f} {sd:>8.4f} {ci_str:>22} {p:>8.4f}")
    results["per_fold"]["pooled"] = {
        "n": len(arr), "mean": mean, "sd": sd, "lo": lo, "hi": hi, "perm_p": p,
    }

    # Nadeau-Bengio corrected t-tests for completeness
    try:
        import scipy
        print()
        print("Nadeau-Bengio corrected t-tests (k=5):")
        for topic in TOPICS:
            t, p_nb = nadeau_bengio_t(diffs_by_topic[topic], k=5)
            print(f"  {TOPIC_DISPLAY[topic]:<10} t={t:+.3f}  p={p_nb:.4f}")
            results["per_fold"][topic]["nb_t"] = t
            results["per_fold"][topic]["nb_p"] = p_nb
        t, p_nb = nadeau_bengio_t(pooled, k=5)
        print(f"  {'POOLED':<10} t={t:+.3f}  p={p_nb:.4f}  (caveat: NB correction is fold-level; "
              f"pooled across seeds is heuristic)")
        results["per_fold"]["pooled"]["nb_t"] = t
        results["per_fold"]["pooled"]["nb_p"] = p_nb
    except ImportError:
        print("\n(scipy not installed; skipping Nadeau-Bengio t-tests. Bootstrap CIs above are sufficient.)")

    print()
    print("=" * 96)
    print("Comparison with seed-level (n=5 per topic, n=15 pooled) used as approximation in paper v4 draft:")
    print("=" * 96)
    SEED_LEVEL = {
        "Statins": (0.020, (-0.011, 0.052), 0.19),
        "Opiods":  (-0.048, (-0.116, 0.020), 0.31),
        "ADHD":    (0.003, (-0.040, 0.046), 0.81),
        "pooled":  (-0.008, (-0.039, 0.024), 0.60),
    }
    print(f"{'Topic':<10} {'Per-fold mean':>14} {'Seed-level mean':>17} {'Diff':>8} {'Per-fold CI':>22} {'Seed-level CI':>22}")
    print("-" * 96)
    for key in ["Statins", "Opiods", "ADHD", "pooled"]:
        r = results["per_fold"].get(key, {})
        if not r:
            continue
        seed = SEED_LEVEL[key]
        diff = r["mean"] - seed[0]
        per_fold_ci = f"[{r['lo']:+.3f}, {r['hi']:+.3f}]"
        seed_ci = f"[{seed[1][0]:+.3f}, {seed[1][1]:+.3f}]"
        print(f"{TOPIC_DISPLAY.get(key, key):<10} {r['mean']:>+14.4f} {seed[0]:>+17.4f} {diff:>+8.4f} {per_fold_ci:>22} {seed_ci:>22}")

    # Save archive
    out_path = OUTPUTS_DIR / "bert_per_fold_bootstrap.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults archived to: {out_path}")
    print()
    print("Drop the BERT rows of Table 5 in paper_draft_v4.tex with the per-fold values above.")
    print("Drop the asterisk note on per-fold approximation (it is no longer needed).")


if __name__ == "__main__":
    main()
