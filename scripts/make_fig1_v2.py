"""Generate fig1_gap_forest_v2 with multi-run BoW Statins ribbon and audit BERT values.

Inputs:
  --input    Path to bow_statins_multirun_summary.json
             (default: ./bow_statins_multirun_summary.json)
  --outdir   Directory to write fig1_gap_forest_v2.{pdf,png}
             (default: . )

Hardcoded inputs (sourced from Cohen_BERT_Extension_Results_Consolidation_v2.md §5.2):
  Audit BERT per-topic and pooled point estimates and bootstrap 95% CIs.

For the canonical regeneration path inside the pipeline that loads BERT values
from analysis_results_full_v2.json instead of hardcoding them, see the patched
make_paper_artifacts.py (CU 178 §6 patch, pending).

Usage:
  python make_fig1_v2.py
  python make_fig1_v2.py --input outputs/bow_statins_multirun_summary.json --outdir figures/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Audit BERT values (Cohen_BERT_Extension_Results_Consolidation_v2.md §5.2)
# ---------------------------------------------------------------------------

# Format: (label, n, mean, ci_lo, ci_hi)
BERT_ROWS = [
    ("Statins (n=5)",   5,  +0.0022, -0.0579, +0.0817),
    ("Opioids (n=5)",   5,  -0.0733, -0.1905, +0.0232),
    ("ADHD (n=5)",      5,  -0.0550, -0.1474, +0.0062),
    ("Pooled (n=15)",  15,  -0.0420, -0.0975, +0.0078),
]

# Original published Statins gap (thesis-era, single-seed). Paper text cites
# both this and the multi-run mean; rendered as a secondary marker.
BOW_STATINS_PUBLISHED = 0.121


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", type=Path,
                   default=Path("bow_statins_multirun_summary.json"),
                   help="Path to bow_statins_multirun_summary.json")
    p.add_argument("--outdir", type=Path, default=Path("."),
                   help="Directory to write fig1_gap_forest_v2.{pdf,png}")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(bow_summary, outdir):
    bow_mean = bow_summary["gap_mean"]
    bow_min = bow_summary["gap_min"]
    bow_max = bow_summary["gap_max"]
    bow_n_runs = bow_summary["n_runs"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(6.8, 3.8))

    labels = [r[0] for r in BERT_ROWS]
    means = [r[2] for r in BERT_ROWS]
    los = [r[3] for r in BERT_ROWS]
    his = [r[4] for r in BERT_ROWS]
    y = np.arange(len(labels))[::-1]

    # Multi-run BoW Statins ribbon (background)
    ax.axvspan(bow_min, bow_max, color="#bcbcbc", alpha=0.35, zorder=0)
    ax.axvline(bow_mean, color="#555555", linestyle="--",
               linewidth=1.1, zorder=1)
    # Published reference (secondary)
    ax.axvline(BOW_STATINS_PUBLISHED, color="#888888", linestyle=":",
               linewidth=1.0, zorder=1)

    # Null reference
    ax.axvline(0, color="black", linestyle="-", linewidth=0.5, zorder=2)

    # BERT CI bars and point estimates
    for yi, mi, lo, hi, lab in zip(y, means, los, his, labels):
        ax.plot([lo, hi], [yi, yi], color="black", linewidth=1.3, zorder=3)
        ax.plot([lo, lo], [yi - 0.12, yi + 0.12], color="black", linewidth=1.3, zorder=3)
        ax.plot([hi, hi], [yi - 0.12, yi + 0.12], color="black", linewidth=1.3, zorder=3)
        if lab.startswith("Pooled"):
            ax.plot(mi, yi, marker="D", color="black", markersize=8,
                    markerfacecolor="black", zorder=4)
        else:
            ax.plot(mi, yi, marker="o", color="black", markersize=6,
                    markerfacecolor="white", markeredgewidth=1.3, zorder=4)

    # Annotate point estimates
    for yi, mi, hi in zip(y, means, his):
        ax.text(hi + 0.006, yi, f"{mi:+.3f}", fontsize=8,
                va="center", color="black")

    # Reference labels in bottom margin
    ribbon_x = (bow_min + bow_max) / 2
    ax.text(ribbon_x, -0.65,
            f"BoW multi-run\n[+{bow_min:.3f}, +{bow_max:.3f}]\nmean +{bow_mean:.3f}",
            fontsize=7.5, color="#333333", ha="center", va="center")
    ax.text(BOW_STATINS_PUBLISHED + 0.003, -1.15,
            f"BoW published (+{BOW_STATINS_PUBLISHED:.3f})",
            fontsize=7.5, color="#555555", ha="left", va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Expert MeSH \u2212 auto MeSH (WSS@95%)")
    ax.set_xlim(-0.22, 0.17)
    ax.set_ylim(-1.5, len(labels) - 0.2)
    ax.set_title(
        "Per-fold expert-vs-auto WSS@95% difference: BiomedBERT vs BoW Statins reference",
        fontsize=10.5, pad=8,
    )
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.tick_params(axis="both", direction="out", length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / "fig1_gap_forest_v2.pdf"
    png_path = outdir / "fig1_gap_forest_v2.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main():
    args = parse_args()
    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with args.input.open() as f:
        bow_summary = json.load(f)["summary"]

    pdf_path, png_path = make_figure(bow_summary, args.outdir)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
