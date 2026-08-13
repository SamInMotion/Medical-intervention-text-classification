"""Forest plot of expert-vs-auto MeSH WSS@95% gap by classifier and topic.

Reads multi-run BoW and multi-seed BERT summary JSONs and renders a forest plot
with per-topic confidence intervals for each classifier. Replaces the v2 figure
(fig1_gap_forest_v2) which used single-run BoW values at Opioids and ADHD.

Usage from repo root:
    python scripts/make_fig1_gap_forest_v3.py

Inputs (in outputs/):
    bow_statins_multirun_summary.json
    bow_opiods_multirun_summary.json
    bow_adhd_multirun_summary.json
    bert_statins_multiseed_summary.json
    bert_opiods_multiseed_summary.json
    bert_adhd_multiseed_summary.json

Outputs:
    outputs/fig1_gap_forest_v3.pdf
    outputs/fig1_gap_forest_v3.png

Dependencies: numpy, matplotlib (both already in the .venv312 environment).
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Configuration -------------------------------------------------------------

TOPICS = ["Statins", "Opiods", "ADHD"]           # cache spelling for filenames
TOPIC_DISPLAY = {"Statins": "Statins", "Opiods": "Opioids", "ADHD": "ADHD"}
TOPIC_N = {"Statins": (2744, 152), "Opiods": (1772, 43), "ADHD": (803, 83)}

OUTPUTS_DIR = Path("outputs")
N_BOOT = 10000
SEED = 42

# Aesthetics chosen for legibility in print and on-screen.
BOW_COLOR = "#4477AA"
BERT_COLOR = "#EE6677"
ZERO_COLOR = "#888888"
PUBLISHED_COLOR = "#333333"
BG_COLOR = "#FFFFFF"

# Reference values to overlay on the Statins row.
PUBLISHED_STATINS_GAP = 0.121  # Cohen et al. (2006) BoW single-run reference

# Helpers -------------------------------------------------------------------

def bootstrap_ci(values, n_boot=N_BOOT, alpha=0.05, seed=SEED):
    """Percentile bootstrap CI on the mean. Returns (lo, hi)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return lo, hi


def load_bow_fold_diffs(topic):
    """Returns list of per-fold expert-auto WSS gaps across all multi-run reruns."""
    path = OUTPUTS_DIR / f"bow_{topic.lower()}_multirun_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    diffs = []
    for run in payload["runs"]:
        modes = run["modes"]
        expert = modes["title_abstract_mesh"]["wss"] if isinstance(modes["title_abstract_mesh"], dict) else modes["title_abstract_mesh"]
        auto   = modes["auto_mesh"]["wss"]            if isinstance(modes["auto_mesh"], dict)            else modes["auto_mesh"]
        for e, a in zip(expert, auto):
            if e is not None and a is not None:
                diffs.append(e - a)
    return diffs


def load_bert_fold_diffs(topic):
    """Returns list of per-fold expert-auto WSS gaps across all multi-seed runs."""
    path = OUTPUTS_DIR / f"bert_{topic.lower()}_multiseed_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    diffs = []
    expert_runs = payload["expert_runs"]
    auto_runs   = payload["auto_runs"]
    for seed in expert_runs:
        if seed not in auto_runs:
            continue
        expert_folds = expert_runs[seed]["folds"]
        auto_folds   = auto_runs[seed]["folds"]
        for e_fold, a_fold in zip(expert_folds, auto_folds):
            e_wss = e_fold.get("wss_at_95")
            a_wss = a_fold.get("wss_at_95")
            if e_wss is not None and a_wss is not None:
                diffs.append(e_wss - a_wss)
    return diffs


def summarise(diffs):
    """Returns dict with mean, sd, lo (95% bootstrap), hi (95% bootstrap), n."""
    diffs = np.asarray(diffs, dtype=float)
    lo, hi = bootstrap_ci(diffs)
    return {
        "mean": float(diffs.mean()),
        "sd":   float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
        "lo":   float(lo),
        "hi":   float(hi),
        "n":    int(len(diffs)),
    }


# Main routine --------------------------------------------------------------

def main():
    summaries = {}
    for topic in TOPICS:
        bow_diffs = load_bow_fold_diffs(topic)
        bert_diffs = load_bert_fold_diffs(topic)
        summaries[topic] = {
            "bow": summarise(bow_diffs),
            "bert": summarise(bert_diffs),
        }

    # Print to stdout for verification before saving the figure.
    print("=" * 86)
    print("Forest plot inputs (expert - auto MeSH WSS@95% gap)")
    print("=" * 86)
    print(f"{'Topic':<10} {'Classifier':<8} {'n':>4} {'Mean':>8} {'SD':>8} {'95% CI':>22}")
    print("-" * 86)
    for topic in TOPICS:
        for clf in ["bow", "bert"]:
            s = summaries[topic][clf]
            ci = f"[{s['lo']:+.3f}, {s['hi']:+.3f}]"
            print(f"{TOPIC_DISPLAY[topic]:<10} {clf.upper():<8} {s['n']:>4} {s['mean']:>+8.3f} {s['sd']:>8.3f} {ci:>22}")
    print()

    # Figure construction.
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Vertical reference at zero (no gap).
    ax.axvline(0.0, color=ZERO_COLOR, linewidth=1.0, linestyle="--", zorder=1)

    # Y positions for topic rows. Three topics, two markers per row (BoW + BERT).
    topic_y = {topic: i for i, topic in enumerate(reversed(TOPICS))}
    offset = 0.16

    for topic in TOPICS:
        y = topic_y[topic]

        # BoW
        s = summaries[topic]["bow"]
        ax.errorbar(s["mean"], y + offset,
                    xerr=[[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]],
                    fmt="o", markersize=8, capsize=4, capthick=1.2,
                    color=BOW_COLOR, ecolor=BOW_COLOR, elinewidth=1.4,
                    zorder=3, label=None)

        # BERT
        s = summaries[topic]["bert"]
        ax.errorbar(s["mean"], y - offset,
                    xerr=[[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]],
                    fmt="s", markersize=8, capsize=4, capthick=1.2,
                    color=BERT_COLOR, ecolor=BERT_COLOR, elinewidth=1.4,
                    zorder=3, label=None)

    # Published Cohen et al. reference point on the Statins row only.
    statins_y = topic_y["Statins"]
    ax.scatter(PUBLISHED_STATINS_GAP, statins_y + offset, marker="x",
               s=72, color=PUBLISHED_COLOR, linewidths=1.6, zorder=4)
    ax.annotate("Cohen et al. (2006)\nBoW single run +0.121",
                xy=(PUBLISHED_STATINS_GAP, statins_y + offset),
                xytext=(PUBLISHED_STATINS_GAP + 0.015, statins_y + offset + 0.22),
                fontsize=8, color=PUBLISHED_COLOR,
                arrowprops=dict(arrowstyle="->", color=PUBLISHED_COLOR, lw=0.7,
                                connectionstyle="arc3,rad=0.15"))

    # Axis labels and ticks.
    ax.set_yticks(list(topic_y.values()))
    labels = []
    for topic in reversed(TOPICS):
        n_abs, n_pos = TOPIC_N[topic]
        labels.append(f"{TOPIC_DISPLAY[topic]}\n({n_pos}/{n_abs})")
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_ylim(-0.6, len(TOPICS) - 0.4)

    ax.set_xlabel("Expert − auto MeSH WSS@95% gap (95% bootstrap CI)", fontsize=10)
    ax.set_xlim(-0.15, 0.20)
    ax.set_xticks(np.arange(-0.15, 0.21, 0.05))
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, color="#cccccc", zorder=0)

    # Legend.
    handles = [
        mpatches.Patch(facecolor=BOW_COLOR, label="BoW multi-run (7 runs × 5 folds = 35 folds)"),
        mpatches.Patch(facecolor=BERT_COLOR, label="BiomedBERT multi-seed (5 seeds × 5 folds = 25 folds)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=False)

    # Title.
    ax.set_title(
        "Cohen benchmark expert-vs-auto MeSH gap by classifier and topic",
        fontsize=11, pad=12, loc="left",
    )

    # The textual summary lives in the paper caption, not on the figure itself.

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUTS_DIR / "fig1_gap_forest_v3.pdf"
    png_path = OUTPUTS_DIR / "fig1_gap_forest_v3.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
