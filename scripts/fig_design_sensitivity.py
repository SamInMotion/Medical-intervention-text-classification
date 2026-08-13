#!/usr/bin/env python3
"""
fig_design_sensitivity.py

Generate the design-sensitivity figure for the NEJLT submission.
Requires: matplotlib, numpy

Usage:
    python scripts/fig_design_sensitivity.py     (run from the repository root)

Output:
    outputs/fig_design_sensitivity_final.pdf
    outputs/fig_design_sensitivity_final.png

CHANGES FROM THE PREVIOUS VERSION (X2)
  F1  The right panel headed its first column "Evaluation design" and listed
      BiomedBERT under it. BiomedBERT at 5-fold IS the canonical evaluation
      design; what changes in that column is the classifier. Header is now
      "Condition", the panel title no longer says every row is a design, and
      the BiomedBERT row states that the design is canonical.
  F2  The trend line ran across all four points, which reads as one monotone
      progression through nested designs. It now spans the three bag-of-words
      conditions only, and a divider separates the classifier substitution.
  F3  The reduction block reported "-79% BiomedBERT" as a gap reduction
      against the canonical BoW reference, presenting a classifier change as a
      design effect. It now covers the two design changes, and the BiomedBERT
      comparison is stated as what the manuscript actually claims: a point
      estimate 0.001 from the BoW 10-fold result.
  F4  Data provenance comment corrected. The values are current as of v7.7,
      not v7.1, and the BiomedBERT interval is the per-fold set (ledger #20),
      not the seed-level set that make_fig2_design_sensitivity.py still holds.
  F5  Removed an unused import; print statements now name the paths written.

NOTE ON THE VALUES
  These are hardcoded. They match Table `tab:design_sensitivity` and the
  BiomedBERT row of `tab:stats` in paper_draft_v7_7. The generating sources
  are paper_experiments/outputs/bow_experiments_summary.csv (rows 2 and 3),
  outputs/bow_statins_multirun_summary.json (row 1) and
  outputs/bert_statins_multiseed_summary.json (row 4). Any edit to those
  tables must be mirrored here; there is no automatic link.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

# ============================================================
# DATA — matches paper_draft_v7_7, tab:design_sensitivity and tab:stats
# ============================================================

# Three bag-of-words evaluation designs, then one classifier substitution
# at the canonical design. The fourth column is not a design change.
designs_short = ['Canonical\nBoW', 'Subsampled\nBoW', '10-fold\nBoW', 'BiomedBERT\n(canonical)']
N_BOW = 3  # first N_BOW columns are bag-of-words evaluation designs

# Mean gaps (WSS@95%)
means = [0.0957, 0.0332, 0.0207, 0.0200]

# 95% bootstrap CIs
ci_lower = [0.075, -0.022, 0.001, -0.021]
ci_upper = [0.116, 0.092, 0.041, 0.062]

# Colors for each condition
colors = ['#1a5276', '#1e8449', '#d68910', '#7d3c98']
light_colors = ['#d6eaf8', '#d5f5e3', '#fef9e7', '#f5eef8']

# ============================================================
# FIGURE SETUP
# ============================================================

fig = plt.figure(figsize=(13.5, 6.8))

ax1 = fig.add_axes([0.06, 0.11, 0.50, 0.80])
ax2 = fig.add_axes([0.61, 0.11, 0.36, 0.80])

# ============================================================
# LEFT PANEL
# ============================================================

x = np.arange(len(designs_short))

for i in range(len(designs_short)):
    ax1.errorbar(
        x[i], means[i],
        yerr=[[means[i] - ci_lower[i]], [ci_upper[i] - means[i]]],
        fmt='o', markersize=11, capsize=4, capthick=2.5,
        color=colors[i], ecolor=colors[i], elinewidth=2,
        markeredgecolor='white', markeredgewidth=2,
        zorder=5
    )

    ax1.annotate(
        f'{means[i]:.3f}',
        xy=(x[i], means[i]),
        xytext=(0, 28),
        textcoords='offset points',
        ha='center', va='bottom',
        fontsize=12, fontweight='bold', color=colors[i],
        zorder=10,
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                  edgecolor='none', alpha=0.85)
    )

    y_label = ci_lower[i] if ci_lower[i] > -0.032 else -0.032
    ax1.annotate(
        f'[{ci_lower[i]:+.3f}, {ci_upper[i]:+.3f}]',
        xy=(x[i], y_label),
        xytext=(0, -24),
        textcoords='offset points',
        ha='center', va='top',
        fontsize=8.5, color='#777777',
        zorder=10,
        bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                  edgecolor='none', alpha=0.8)
    )

# F2: trend line across the bag-of-words designs only. Extending it to the
# fourth point implies a progression through nested designs that does not
# exist; the fourth point is a different classifier at the first design.
ax1.plot(x[:N_BOW], means[:N_BOW], 'k-', linewidth=1, alpha=0.18, zorder=1)

# F2: divider marking where the manipulation stops being the design
ax1.axvline(x=N_BOW - 0.5, color='#999999', linestyle=':', linewidth=1.0,
            alpha=0.7, zorder=1)
ax1.text(N_BOW - 0.46, 0.131, 'classifier change,\ncanonical design',
         fontsize=7.5, color='#777777', ha='left', va='top', linespacing=1.25)
ax1.text(N_BOW - 0.54, 0.131, 'evaluation design varied',
         fontsize=7.5, color='#777777', ha='right', va='top')

ax1.axhline(y=0, color='#c0392b', linestyle='--', linewidth=0.9, alpha=0.5, zorder=2)
ax1.text(3.15, 0.003, 'zero', fontsize=8.5, color='#c0392b', alpha=0.6, va='bottom')

for i in range(len(designs_short)):
    if ci_lower[i] < 0 < ci_upper[i]:
        ax1.axhspan(
            max(ci_lower[i], -0.04), 0,
            xmin=(x[i] - 0.1 + 0.3) / 3.6, xmax=(x[i] + 0.1 + 0.3) / 3.6,
            alpha=0.05, color='#c0392b', zorder=0
        )

ax1.set_xticks(x)
ax1.set_xticklabels(designs_short, fontsize=10)
ax1.set_ylabel('Expert – Auto WSS@95% difference', fontsize=11, fontweight='bold')

ax1.set_ylim(-0.04, 0.15)
ax1.set_xlim(-0.3, 3.3)

ax1.yaxis.grid(True, linestyle='--', alpha=0.25)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax1.set_title(
    'Statins expert–auto MeSH gap by evaluation design and classifier',
    fontsize=12.5, fontweight='bold', pad=12
)

# ============================================================
# RIGHT PANEL
# ============================================================

ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# F1: the panel no longer asserts that every row is an evaluation design
ax2.text(5, 9.7, 'Statins expert–vs–auto MeSH gap',
         ha='center', va='top', fontsize=11.5, fontweight='bold', color='#1a1a1a')
ax2.text(5, 9.2, 'by condition',
         ha='center', va='top', fontsize=11.5, fontweight='bold', color='#1a1a1a')

header_y = 8.35
ax2.add_patch(Rectangle((0.15, header_y - 0.38), 9.7, 0.76,
                        facecolor='#1a5276', edgecolor='none'))
ax2.text(1.4, header_y, 'Condition', ha='left', va='center',
         fontsize=8.5, fontweight='bold', color='white')
ax2.text(5.7, header_y, 'Gap', ha='center', va='center',
         fontsize=8.5, fontweight='bold', color='white')
ax2.text(8.3, header_y, '95% CI', ha='center', va='center',
         fontsize=8.5, fontweight='bold', color='white')

# F1: the BiomedBERT row states that its design is the canonical one
row_data = [
    ('Canonical BoW\n(5-fold, full n=2,744)', '0.096', '[+0.075, +0.116]', colors[0], light_colors[0]),
    ('Subsampled BoW\n(5-fold, n=803)', '0.033', '[-0.022, +0.092]', colors[1], light_colors[1]),
    ('10-fold BoW\n(10-fold, full n=2,744)', '0.021', '[+0.001, +0.041]', colors[2], light_colors[2]),
    ('BiomedBERT\n(canonical design, classifier change)', '0.020', '[-0.021, +0.062]', colors[3], light_colors[3]),
]

row_y = [7.15, 5.75, 4.35, 2.95]
for i, (design, gap, ci, color, bg) in enumerate(row_data):
    y = row_y[i]
    ax2.add_patch(Rectangle((0.15, y - 0.52), 9.7, 1.04,
                            facecolor=bg, edgecolor='#e5e5e5', linewidth=0.5))
    ax2.add_patch(Rectangle((0.35, y - 0.2), 0.38, 0.4,
                            facecolor=color, edgecolor='none', alpha=0.9))

    ax2.text(0.9, y, design, ha='left', va='center',
             fontsize=7.6, color='#333333', linespacing=1.1)
    ax2.text(5.7, y, gap, ha='center', va='center',
             fontsize=11, fontweight='bold', color=color)
    ax2.text(8.3, y, ci, ha='center', va='center',
             fontsize=7.8, color='#555555')

# F3: reduction block covers the two design changes only. The BiomedBERT
# comparison is stated as the manuscript states it.
red_y = 1.45
ax2.add_patch(FancyBboxPatch((0.15, red_y - 1.05), 9.7, 1.25,
                             boxstyle="round,pad=0.02,rounding_size=0.15",
                             facecolor='#fafafa', edgecolor='#bbbbbb',
                             linewidth=0.8))

ax2.text(5, red_y + 0.35, 'Gap change vs. canonical BoW, evaluation design only',
         ha='center', va='center', fontsize=8.5, fontweight='bold', color='#444444')

reductions = [
    ('Subsampled', '–65%', colors[1]),
    ('10-fold', '–78%', colors[2]),
]

for i, (label, pct, color) in enumerate(reductions):
    x_pos = 2.4 + i * 2.4
    ax2.text(x_pos, red_y - 0.12, pct, ha='center', va='center',
             fontsize=12, fontweight='bold', color=color)
    ax2.text(x_pos, red_y - 0.44, label, ha='center', va='center',
             fontsize=7.5, color='#888888')

ax2.plot([3.3, 3.9], [red_y - 0.12, red_y - 0.12], 'k-', linewidth=0.4, alpha=0.15)

ax2.text(7.9, red_y - 0.12, '0.001', ha='center', va='center',
         fontsize=12, fontweight='bold', color=colors[3])
ax2.text(7.9, red_y - 0.44, 'BiomedBERT vs 10-fold BoW', ha='center', va='center',
         fontsize=7.5, color='#888888')
ax2.plot([6.05, 6.05], [red_y - 0.62, red_y + 0.18], color='#cccccc',
         linewidth=0.8, linestyle=':')

ax2.text(5, 0.15, 'Bootstrap percentile CIs, 10,000 resamples',
         ha='center', va='center', fontsize=7, color='#aaaaaa', style='italic')

# ============================================================
# SAVE
# ============================================================

PDF = 'outputs/fig_design_sensitivity_final.pdf'
PNG = 'outputs/fig_design_sensitivity_final.png'
plt.savefig(PDF, dpi=300, facecolor='white', bbox_inches='tight')
plt.savefig(PNG, dpi=300, facecolor='white', bbox_inches='tight')
print(f"Saved: {PDF}")
print(f"Saved: {PNG}")
