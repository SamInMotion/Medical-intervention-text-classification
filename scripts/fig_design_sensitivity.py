#!/usr/bin/env python3
"""
fig_design_sensitivity.py

Generate the design-sensitivity figure for the NEJLT submission.
Requires: matplotlib, numpy

Usage:
    python fig_design_sensitivity.py

Output:
    fig_design_sensitivity_final.pdf
    fig_design_sensitivity_final.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

# ============================================================
# DATA — from manuscript v7.1 (NEJLT submission)
# ============================================================

designs_short = ['Canonical\nBoW', 'Subsampled\nBoW', '10-fold\nBoW', 'BiomedBERT']

# Mean gaps (WSS@95%)
means = [0.0957, 0.0332, 0.0207, 0.0200]

# 95% bootstrap CIs from manuscript
ci_lower = [0.075, -0.022, 0.001, -0.011]
ci_upper = [0.116, 0.092, 0.041, 0.052]

# Colors for each design condition
colors = ['#1a5276', '#1e8449', '#d68910', '#7d3c98']
light_colors = ['#d6eaf8', '#d5f5e3', '#fef9e7', '#f5eef8']

# ============================================================
# FIGURE SETUP
# ============================================================

fig = plt.figure(figsize=(13.5, 6.8))

# Left panel: main plot | Right panel: summary table
# [left, bottom, width, height] in figure coordinates
ax1 = fig.add_axes([0.06, 0.11, 0.50, 0.80])
ax2 = fig.add_axes([0.61, 0.11, 0.36, 0.80])

# ============================================================
# LEFT PANEL: Line plot with error bars
# ============================================================

x = np.arange(len(designs_short))

for i in range(len(designs_short)):
    # Error bar (vertical line with caps)
    ax1.errorbar(
        x[i], means[i],
        yerr=[[means[i] - ci_lower[i]], [ci_upper[i] - means[i]]],
        fmt='o', markersize=11, capsize=4, capthick=2.5,
        color=colors[i], ecolor=colors[i], elinewidth=2,
        markeredgecolor='white', markeredgewidth=2,
        zorder=5
    )

    # Value label above point — with white bbox to prevent overlap
    ax1.annotate(
        f'{means[i]:.3f}',
        xy=(x[i], means[i]),
        xytext=(0, 28),           # vertical offset (points)
        textcoords='offset points',
        ha='center', va='bottom',
        fontsize=12, fontweight='bold', color=colors[i],
        zorder=10,                # draw ON TOP of error bars
        bbox=dict(
            boxstyle='round,pad=0.15',
            facecolor='white',
            edgecolor='none',
            alpha=0.85
        )
    )

    # CI label below error bar — with white bbox
    y_label = ci_lower[i] if ci_lower[i] > -0.032 else -0.032
    ax1.annotate(
        f'[{ci_lower[i]:+.3f}, {ci_upper[i]:+.3f}]',
        xy=(x[i], y_label),
        xytext=(0, -24),          # vertical offset (points)
        textcoords='offset points',
        ha='center', va='top',
        fontsize=8.5, color='#777777',
        zorder=10,
        bbox=dict(
            boxstyle='round,pad=0.1',
            facecolor='white',
            edgecolor='none',
            alpha=0.8
        )
    )

# Trend line connecting points
ax1.plot(x, means, 'k-', linewidth=1, alpha=0.18, zorder=1)

# Zero reference line (red dashed)
ax1.axhline(y=0, color='#c0392b', linestyle='--', linewidth=0.9, alpha=0.5, zorder=2)
ax1.text(3.15, 0.003, 'zero', fontsize=8.5, color='#c0392b', alpha=0.6, va='bottom')

# Subtle red shading where CIs cross zero (subsampled + BiomedBERT)
for i in [1, 3]:
    if ci_lower[i] < 0 < ci_upper[i]:
        ax1.axhspan(
            max(ci_lower[i], -0.04), 0,
            xmin=(x[i] - 0.1) / 3.2, xmax=(x[i] + 0.1) / 3.2,
            alpha=0.05, color='#c0392b', zorder=0
        )

# Axis formatting
ax1.set_xticks(x)
ax1.set_xticklabels(designs_short, fontsize=10)
ax1.set_ylabel('Expert – Auto WSS@95% difference', fontsize=11, fontweight='bold')

ax1.set_ylim(-0.04, 0.14)      # headroom for labels
ax1.set_xlim(-0.3, 3.3)

ax1.yaxis.grid(True, linestyle='--', alpha=0.25)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax1.set_title(
    'Design sensitivity of the Statins expert–auto MeSH gap',
    fontsize=12.5, fontweight='bold', pad=12
)

# ============================================================
# RIGHT PANEL: Summary table
# ============================================================

ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Title
ax2.text(5, 9.7, 'Effect of evaluation design on',
         ha='center', va='top', fontsize=11.5, fontweight='bold', color='#1a1a1a')
ax2.text(5, 9.2, 'expert–vs–auto MeSH gap',
         ha='center', va='top', fontsize=11.5, fontweight='bold', color='#1a1a1a')

# Table header
header_y = 8.35
ax2.add_patch(Rectangle((0.15, header_y - 0.38), 9.7, 0.76,
                        facecolor='#1a5276', edgecolor='none'))
ax2.text(1.4, header_y, 'Evaluation design', ha='left', va='center',
         fontsize=8.5, fontweight='bold', color='white')
ax2.text(5.7, header_y, 'Gap', ha='center', va='center',
         fontsize=8.5, fontweight='bold', color='white')
ax2.text(8.3, header_y, '95% CI', ha='center', va='center',
         fontsize=8.5, fontweight='bold', color='white')

# Table rows
row_data = [
    ('Canonical BoW\n(5-fold, full n=2,744)', '0.096', '[+0.075, +0.116]', colors[0], light_colors[0]),
    ('Subsampled BoW\n(5-fold, n=803)', '0.033', '[-0.022, +0.092]', colors[1], light_colors[1]),
    ('10-fold BoW\n(10-fold, full n=2,744)', '0.021', '[+0.001, +0.041]', colors[2], light_colors[2]),
    ('BiomedBERT\n(5-fold, truncated 512)', '0.020', '[-0.011, +0.052]', colors[3], light_colors[3]),
]

row_y = [7.15, 5.75, 4.35, 2.95]
for i, (design, gap, ci, color, bg) in enumerate(row_data):
    y = row_y[i]
    ax2.add_patch(Rectangle((0.15, y - 0.52), 9.7, 1.04,
                            facecolor=bg, edgecolor='#e5e5e5', linewidth=0.5))
    ax2.add_patch(Rectangle((0.35, y - 0.2), 0.38, 0.4,
                            facecolor=color, edgecolor='none', alpha=0.9))

    ax2.text(0.9, y, design, ha='left', va='center',
             fontsize=8, color='#333333', linespacing=1.1)
    ax2.text(5.7, y, gap, ha='center', va='center',
             fontsize=11, fontweight='bold', color=color)
    ax2.text(8.3, y, ci, ha='center', va='center',
             fontsize=7.8, color='#555555')

# Reduction summary box
red_y = 1.45
ax2.add_patch(FancyBboxPatch((0.15, red_y - 1.05), 9.7, 1.25,
                             boxstyle="round,pad=0.02,rounding_size=0.15",
                             facecolor='#fafafa', edgecolor='#bbbbbb',
                             linewidth=0.8))

ax2.text(5, red_y + 0.35, 'Gap reduction vs. Canonical BoW',
         ha='center', va='center', fontsize=8.5, fontweight='bold', color='#444444')

reductions = [
    ('Subsampled', '–65%', colors[1]),
    ('10-fold', '–78%', colors[2]),
    ('BiomedBERT', '–79%', colors[3]),
]

for i, (label, pct, color) in enumerate(reductions):
    x_pos = 1.7 + i * 2.6
    ax2.text(x_pos, red_y - 0.15, pct, ha='center', va='center',
             fontsize=12, fontweight='bold', color=color)
    ax2.text(x_pos, red_y - 0.48, label, ha='center', va='center',
             fontsize=7.5, color='#888888')
    if i < 2:
        ax2.plot([x_pos + 0.7, x_pos + 1.6], [red_y - 0.15, red_y - 0.15],
                'k-', linewidth=0.4, alpha=0.15)

# Footer note
ax2.text(5, 0.15, 'Bootstrap percentile CIs, 10,000 resamples',
         ha='center', va='center', fontsize=7, color='#aaaaaa', style='italic')

# ============================================================
# SAVE
# ============================================================

plt.savefig('fig_design_sensitivity_final.pdf', dpi=300, facecolor='white', bbox_inches='tight')
plt.savefig('fig_design_sensitivity_final.png', dpi=300, facecolor='white', bbox_inches='tight')
print("Saved: fig_design_sensitivity_final.pdf")
print("Saved: fig_design_sensitivity_final.png")
