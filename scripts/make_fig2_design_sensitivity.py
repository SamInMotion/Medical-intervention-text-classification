"""
Figure 2: Design sensitivity of the Statins expert-vs-auto MeSH gap.

Produces fig2_design_sensitivity.{pdf,png} showing four Statins conditions
(canonical BoW, subsampled BoW, 10-fold BoW, canonical BERT) plus Opioids/ADHD
multi-run references for cross-topic comparison.

Uses colour-blind-safe palette (matches make_fig1_gap_forest_v3.py convention).
No matplotlib style dependencies; standalone.

Usage:
    python make_fig2_design_sensitivity.py

Output:
    fig2_design_sensitivity.pdf
    fig2_design_sensitivity.png (300 DPI)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Rows top-to-bottom (plot from bottom-up so top row displays at top)
# (label, mean, ci_lo, ci_hi, group)
# group: 'canonical_bow' | 'design_perturbation' | 'bert' | 'reference'
ROWS = [
    ("BoW canonical\n(Statins, $n$=2,744, 5-fold)",     +0.0957, +0.075, +0.116, "canonical_bow"),
    ("BoW subsampled\n(Statins, $n$=803, 5-fold)",      +0.0332, -0.022, +0.092, "design_perturbation"),
    ("BoW 10-fold\n(Statins, $n$=2,744, 10-fold)",      +0.0207, +0.001, +0.041, "design_perturbation"),
    ("BiomedBERT canonical\n(Statins, $n$=2,744, 5-fold)", +0.020,  -0.011, +0.052, "bert"),
    ("BoW multi-run\n(Opioids, $n$=1,772, 5-fold)",     +0.0066, -0.050, +0.061, "reference"),
    ("BoW multi-run\n(ADHD, $n$=803, 5-fold)",          +0.0059, -0.046, +0.060, "reference"),
]

# Colour-blind-safe palette
COLOURS = {
    "canonical_bow":        "#D55E00",  # vermillion — the reference the paper argues against
    "design_perturbation":  "#0072B2",  # blue — the design-sensitivity result
    "bert":                 "#009E73",  # bluish-green — the canonical-design cross-classifier result
    "reference":            "#999999",  # grey — cross-topic references
}


def make_figure():
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    n_rows = len(ROWS)
    y_positions = list(range(n_rows, 0, -1))  # top row at highest y

    for y, (label, mean, ci_lo, ci_hi, group) in zip(y_positions, ROWS):
        colour = COLOURS[group]
        ax.errorbar(
            mean, y,
            xerr=[[mean - ci_lo], [ci_hi - mean]],
            fmt='o',
            color=colour,
            ecolor=colour,
            elinewidth=2.0,
            capsize=4,
            markersize=8,
            markeredgecolor='black',
            markeredgewidth=0.6,
        )

    # Reference line at zero
    ax.axvline(x=0.0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

    # Y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row[0] for row in ROWS], fontsize=9)

    # X-axis
    ax.set_xlabel("Expert − auto MeSH gap (WSS@95%)", fontsize=11)
    ax.set_xlim(-0.10, 0.14)
    ax.set_xticks(np.arange(-0.10, 0.16, 0.02))
    ax.tick_params(axis='x', labelsize=9)

    # Grid
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Y-limits with padding
    ax.set_ylim(0.4, n_rows + 0.6)

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLOURS["canonical_bow"], label="Canonical BoW (reference)"),
        mpatches.Patch(color=COLOURS["design_perturbation"], label="BoW design perturbation"),
        mpatches.Patch(color=COLOURS["bert"], label="BiomedBERT canonical"),
        mpatches.Patch(color=COLOURS["reference"], label="Cross-topic reference"),
    ]
    ax.legend(
        handles=legend_handles,
        loc='lower right',
        fontsize=8,
        frameon=True,
        framealpha=0.9,
    )

    # Annotations
    ax.text(
        0.0, n_rows + 0.35,
        "← Auto MeSH advantage    |    Expert MeSH advantage →",
        ha='center', va='bottom', fontsize=8, style='italic', color='#555555',
    )

    plt.tight_layout()

    # Save both PDF and PNG
    fig.savefig("fig2_design_sensitivity.pdf", format='pdf', bbox_inches='tight')
    fig.savefig("fig2_design_sensitivity.png", format='png', dpi=300, bbox_inches='tight')

    print("Wrote fig2_design_sensitivity.pdf")
    print("Wrote fig2_design_sensitivity.png (300 DPI)")


if __name__ == "__main__":
    make_figure()
