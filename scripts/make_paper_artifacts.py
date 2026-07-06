"""Generate paper-ready figures, tables, and supplementary CSVs.

Produces:
  figures/
    fig1_gap_forest.pdf    Forest plot of expert-vs-auto WSS@95% difference
                           per topic plus pooled, with BoW reference line.
    fig1_gap_forest.png    Same, raster preview.
    fig2_wss_by_mode.pdf   Per-topic bar chart of WSS@95% across four text
                           modes with fold-level error bars.
    fig2_wss_by_mode.png   Same, raster preview.
  tables/
    table1_topic_characteristics.tex
    table2_bow_statins.tex
    table3_bert_results.tex
    table4_statistical_tests.tex
  supplementary/
    per_fold_wss.csv       Per-fold WSS@95% per topic-mode (long format)
    per_fold_diffs.csv     Per-fold expert-auto differences (long format)

Run from the directory holding the colab_outputs/ folder:
    python make_paper_artifacts.py colab_outputs
"""

import csv
import math
import re
import sys
from itertools import product
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOPICS = ["Statins", "Opiods", "ADHD"]
# Display labels: the Cohen TSV preserves the original "Opiods" spelling;
# the paper standardises on "Opioids". Map filename -> paper label.
TOPIC_LABEL = {"Statins": "Statins", "Opiods": "Opioids", "ADHD": "ADHD"}
MODES = ["abstract", "title_abstract", "title_abstract_mesh", "auto_mesh"]
MODE_LABEL = {
    "abstract": "abstract",
    "title_abstract": "title+abstract",
    "title_abstract_mesh": "expert MeSH",
    "auto_mesh": "auto MeSH",
}
EXPERT_MODE = "title_abstract_mesh"
AUTO_MODE = "auto_mesh"

# BoW Statins reference values from cohen_benchmark_analysis.md
BOW_STATINS = {
    "abstract": 0.123,
    "title_abstract": 0.114,
    "title_abstract_mesh": 0.223,
    "auto_mesh": 0.102,
}
BOW_STATINS_GAP = BOW_STATINS["title_abstract_mesh"] - BOW_STATINS["auto_mesh"]  # 0.121

# Cohen topic characteristics
TOPIC_CHARS = {
    "Statins":  {"n_total": 2744, "n_pos": 152, "n_test": 549, "n_train": 2195},
    "Opiods":   {"n_total": 1772, "n_pos": 43,  "n_test": 354, "n_train": 1418},
    "ADHD":     {"n_total": 803,  "n_pos": 83,  "n_test": 161, "n_train": 642},
}

N_BOOT = 10000
ALPHA = 0.05
SEED = 42


# ---------------------------------------------------------------------------
# Parser (matches bootstrap_paired_permutation.py)
# ---------------------------------------------------------------------------

FOLD_RE = re.compile(
    r"^\s*Fold\s+(\d+):\s+acc=([\-\d\.]+)\s+AUC=([\-\d\.]+)\s+WSS@95=([\-\d\.]+)",
    re.MULTILINE,
)


def parse_txt(path):
    text = Path(path).read_text(encoding="utf-8")
    matches = FOLD_RE.findall(text)
    fold_accs = [float(m[1]) for m in matches]
    fold_aucs = [float(m[2]) for m in matches]
    fold_wsss = [float(m[3]) for m in matches]
    return {"acc": fold_accs, "auc": fold_aucs, "wss": fold_wsss}


def load_all(input_dir):
    """Return data[topic][mode] = {acc, auc, wss} lists of length n_folds."""
    data = {}
    for topic in TOPICS:
        data[topic] = {}
        for mode in MODES:
            path = input_dir / f"bert_{topic.lower()}_{mode}.txt"
            if not path.exists():
                print(f"WARNING: missing {path}", file=sys.stderr)
                continue
            data[topic][mode] = parse_txt(path)
    return data


# ---------------------------------------------------------------------------
# Statistical analysis (matches bootstrap_paired_permutation.py)
# ---------------------------------------------------------------------------

def bootstrap_ci(diffs, n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    return {
        "mean": float(diffs.mean()),
        "ci_lo": float(np.percentile(boot_means, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    }


def paired_perm_p(diffs):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    obs_mean = float(diffs.mean())
    extreme = 0
    total = 2 ** n
    for signs in product([-1, 1], repeat=n):
        perm_mean = float(np.sum(np.asarray(signs) * diffs)) / n
        if abs(perm_mean) >= abs(obs_mean) - 1e-12:
            extreme += 1
    return extreme / total


def nadeau_bengio(diffs, n_train, n_test):
    from scipy import stats
    diffs = np.asarray(diffs, dtype=float)
    k = len(diffs)
    mean_d = float(diffs.mean())
    var_d = float(diffs.var(ddof=1))
    correction = 1.0 / k + n_test / n_train
    se = math.sqrt(var_d * correction)
    t_stat = mean_d / se if se > 0 else float("nan")
    df = k - 1
    p = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df))) if se > 0 else float("nan")
    return {"t_stat": t_stat, "p_value": p}


# ---------------------------------------------------------------------------
# Figure 1: forest plot of expert-vs-auto gap
# ---------------------------------------------------------------------------

def make_fig1(per_topic, pooled, output_dir):
    """Forest plot of expert-auto difference with bootstrap CIs.

    Per topic and pooled, with BoW Statins reference line and null line.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    # y positions: Statins, Opioids, ADHD top to bottom, gap, pooled
    labels = []
    means = []
    los = []
    his = []
    for r in per_topic:
        labels.append(f"{TOPIC_LABEL[r['topic']]} (n={r['n']})")
        means.append(r["mean"])
        los.append(r["ci_lo"])
        his.append(r["ci_hi"])
    labels.append("Pooled (n=15)")
    means.append(pooled["mean"])
    los.append(pooled["ci_lo"])
    his.append(pooled["ci_hi"])

    y = np.arange(len(labels))[::-1]  # top-to-bottom reading order

    # CI bars
    for yi, mi, lo, hi in zip(y, means, los, his):
        ax.plot([lo, hi], [yi, yi], color="black", linewidth=1.3)
        ax.plot([lo, lo], [yi - 0.12, yi + 0.12], color="black", linewidth=1.3)
        ax.plot([hi, hi], [yi - 0.12, yi + 0.12], color="black", linewidth=1.3)
    # Point estimates (pooled emphasised)
    for i, (yi, mi) in enumerate(zip(y, means)):
        if labels[i].startswith("Pooled"):
            ax.plot(mi, yi, marker="D", color="black", markersize=8,
                    markerfacecolor="black")
        else:
            ax.plot(mi, yi, marker="o", color="black", markersize=6,
                    markerfacecolor="white", markeredgewidth=1.3)

    # Null reference
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.6, zorder=0)

    # BoW Statins reference
    ax.axvline(BOW_STATINS_GAP, color="#888888", linestyle="--", linewidth=1.2,
               zorder=0)
    ax.text(BOW_STATINS_GAP - 0.003, 0.3,
            f"BoW Statins\nreference\n(+{BOW_STATINS_GAP:.3f})",
            fontsize=8, color="#444444", va="bottom", ha="right")

    # Annotate point estimates with their values
    for yi, mi, hi in zip(y, means, his):
        ax.text(hi + 0.005, yi, f"{mi:+.3f}", fontsize=8, va="center",
                color="black")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Expert MeSH − auto MeSH (WSS@95%)")
    ax.set_xlim(-0.16, 0.18)
    ax.set_title("Per-fold expert-vs-auto WSS@95% difference with BiomedBERT",
                 fontsize=11, pad=8)

    # subtle x-axis ticks
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.tick_params(axis="both", direction="out", length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    pdf_path = output_dir / "fig1_gap_forest.pdf"
    png_path = output_dir / "fig1_gap_forest.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


# ---------------------------------------------------------------------------
# Figure 2: WSS@95% by mode, per topic (BERT) with BoW Statins reference
# ---------------------------------------------------------------------------

def make_fig2(data, output_dir):
    """Per-topic, per-mode WSS@95% with fold-level error bars.

    Three panels (one per topic), grouped bars (one per mode).
    Statins panel also shows the BoW reference bars in a paler colour.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4), sharey=True)

    mode_xs = np.arange(len(MODES))
    bar_w = 0.32

    for ax, topic in zip(axes, TOPICS):
        bert_means = []
        bert_stds = []
        bow_means = []
        for mode in MODES:
            wss = data[topic][mode]["wss"]
            bert_means.append(np.mean(wss))
            bert_stds.append(np.std(wss, ddof=0))
            bow_means.append(BOW_STATINS[mode] if topic == "Statins" else None)

        # BERT bars (always)
        offset = -bar_w / 2 if topic == "Statins" else 0
        bert_bars = ax.bar(
            mode_xs + offset, bert_means, width=bar_w if topic == "Statins" else bar_w * 1.6,
            yerr=bert_stds, capsize=3,
            color="#3a4a5c", edgecolor="black", linewidth=0.6, label="BiomedBERT",
        )

        # BoW bars (Statins only)
        if topic == "Statins":
            ax.bar(
                mode_xs + bar_w / 2, bow_means, width=bar_w,
                color="#c0c0c0", edgecolor="black", linewidth=0.6, label="BoW LogReg",
            )

        ax.set_title(TOPIC_LABEL[topic], fontsize=11)
        ax.set_xticks(mode_xs)
        ax.set_xticklabels([MODE_LABEL[m] for m in MODES], rotation=30, ha="right",
                           fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", direction="out", length=3)
        ax.set_ylim(-0.05, 0.95)
        ax.axhline(0, color="black", linewidth=0.5)

    axes[0].set_ylabel("WSS@95%")
    # Legend on the Statins panel only (it has both classifiers)
    axes[0].legend(loc="upper left", fontsize=8.5, frameon=False)

    fig.suptitle("WSS@95% by text mode, three Cohen topics", fontsize=11, y=1.02)
    fig.tight_layout()
    pdf_path = output_dir / "fig2_wss_by_mode.pdf"
    png_path = output_dir / "fig2_wss_by_mode.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def latex_table_1(output_dir):
    body = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "Topic & Articles & Included & Inclusion rate \\\\\n"
        "\\midrule\n"
    )
    for topic in TOPICS:
        c = TOPIC_CHARS[topic]
        rate = 100 * c["n_pos"] / c["n_total"]
        body += (f"{TOPIC_LABEL[topic]} & {c['n_total']:,} & {c['n_pos']} "
                 f"& {rate:.1f}\\% \\\\\n")
    body += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Cohen et al.\\ (2006) topic characteristics. "
        "Inclusion rate is the fraction of articles included by the original "
        "systematic review reviewers.}\n"
        "\\label{tab:topics}\n"
        "\\end{table}\n"
    )
    (output_dir / "table1_topic_characteristics.tex").write_text(body)


def latex_table_2(output_dir):
    body = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{lrr}\n"
        "\\toprule\n"
        "Mode & WSS@95\\% & ROC AUC \\\\\n"
        "\\midrule\n"
        "abstract & 0.123 & 0.752 \\\\\n"
        "title+abstract & 0.114 & 0.760 \\\\\n"
        "title+abstract+expert MeSH & \\textbf{0.223} & 0.774 \\\\\n"
        "auto MeSH & 0.102 & 0.749 \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Bag-of-words logistic regression on Cohen Statins (regularised). "
        "Expert MeSH outperforms auto MeSH by 0.121 WSS@95\\%, and auto MeSH "
        "performs below the unaugmented abstract baseline.}\n"
        "\\label{tab:bow_statins}\n"
        "\\end{table}\n"
    )
    (output_dir / "table2_bow_statins.tex").write_text(body)


def latex_table_3(data, output_dir):
    body = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        "Topic & abstract & title+abstract & expert MeSH & auto MeSH "
        "& Expert$-$Auto \\\\\n"
        "\\midrule\n"
    )
    for topic in TOPICS:
        vals = {}
        for mode in MODES:
            wss = data[topic][mode]["wss"]
            vals[mode] = (np.mean(wss), np.std(wss, ddof=0))
        gap = vals[EXPERT_MODE][0] - vals[AUTO_MODE][0]
        gap_str = f"${gap:+.3f}$"
        body += (
            f"{TOPIC_LABEL[topic]} "
            f"& {vals['abstract'][0]:.3f}$\\pm${vals['abstract'][1]:.3f} "
            f"& {vals['title_abstract'][0]:.3f}$\\pm${vals['title_abstract'][1]:.3f} "
            f"& {vals['title_abstract_mesh'][0]:.3f}$\\pm${vals['title_abstract_mesh'][1]:.3f} "
            f"& {vals['auto_mesh'][0]:.3f}$\\pm${vals['auto_mesh'][1]:.3f} "
            f"& {gap_str} \\\\\n"
        )
    body += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{BiomedBERT WSS@95\\% across four text modes on three Cohen "
        "topics, 5-fold stratified cross-validation. Values are fold means "
        "with population standard deviations. The Expert$-$Auto column gives "
        "the per-topic difference between expert MeSH and auto MeSH modes; "
        "the bag-of-words reference value on Statins is $+0.121$.}\n"
        "\\label{tab:bert_results}\n"
        "\\end{table*}\n"
    )
    (output_dir / "table3_bert_results.tex").write_text(body)


def latex_table_4(per_topic, pooled, output_dir):
    body = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{lrrcrr}\n"
        "\\toprule\n"
        "Topic & $k$ & Mean diff & 95\\% CI & Perm $p$ & NB $p$ \\\\\n"
        "\\midrule\n"
    )
    for r in per_topic:
        ci = f"$[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]$"
        nb_p = f"{r['nb_p']:.3f}" if r.get("nb_p") is not None else "n/a"
        body += (
            f"{TOPIC_LABEL[r['topic']]} & {r['n']} & ${r['mean']:+.3f}$ & {ci} "
            f"& {r['perm_p']:.3f} & {nb_p} \\\\\n"
        )
    body += "\\midrule\n"
    ci_p = f"$[{pooled['ci_lo']:+.3f}, {pooled['ci_hi']:+.3f}]$"
    body += (
        f"\\textbf{{Pooled}} & {pooled['n']} & $\\mathbf{{{pooled['mean']:+.3f}}}$ "
        f"& \\textbf{{{ci_p}}} & \\textbf{{{pooled['perm_p']:.3f}}} & n/a \\\\\n"
    )
    body += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Statistical analysis of the expert$-$auto WSS@95\\% "
        "difference. Mean diff is the mean of per-fold paired differences. "
        "95\\% CI is from $10{,}000$-resample percentile bootstrap. "
        "Perm $p$ is from exact paired permutation tests "
        "($2^k$ sign patterns enumerated). NB $p$ is the Nadeau and Bengio "
        "(2003) corrected resampled $t$-test for $k$-fold CV with overlapping "
        "training sets. The pooled bootstrap 95\\% CI is $[-0.052, +0.010]$, "
        "excluding the bag-of-words reference value of $+0.121$ by "
        "approximately 3.6 times the interval half-width.}\n"
        "\\label{tab:stat_tests}\n"
        "\\end{table}\n"
    )
    (output_dir / "table4_statistical_tests.tex").write_text(body)


# ---------------------------------------------------------------------------
# Supplementary CSVs
# ---------------------------------------------------------------------------

def write_per_fold_csv(data, output_dir):
    rows = []
    for topic in TOPICS:
        for mode in MODES:
            if mode not in data[topic]:
                continue
            for i, (acc, auc, wss) in enumerate(
                zip(data[topic][mode]["acc"],
                    data[topic][mode]["auc"],
                    data[topic][mode]["wss"]),
                start=1,
            ):
                rows.append({
                    "topic": TOPIC_LABEL[topic],
                    "mode": mode,
                    "fold": i,
                    "accuracy": acc,
                    "roc_auc": auc,
                    "wss_at_95": wss,
                })
    path = output_dir / "per_fold_wss.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["topic", "mode", "fold", "accuracy", "roc_auc", "wss_at_95"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_per_fold_diffs_csv(data, output_dir):
    rows = []
    for topic in TOPICS:
        if EXPERT_MODE not in data[topic] or AUTO_MODE not in data[topic]:
            continue
        e_wss = data[topic][EXPERT_MODE]["wss"]
        a_wss = data[topic][AUTO_MODE]["wss"]
        for i, (e, a) in enumerate(zip(e_wss, a_wss), start=1):
            rows.append({
                "topic": TOPIC_LABEL[topic],
                "fold": i,
                "expert_wss": round(e, 4),
                "auto_wss": round(a, 4),
                "diff": round(e - a, 4),
            })
    path = output_dir / "per_fold_diffs.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["topic", "fold", "expert_wss", "auto_wss", "diff"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python make_paper_artifacts.py <input_dir>")
        sys.exit(1)
    input_dir = Path(sys.argv[1])
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    out_base = Path("paper_artifacts")
    fig_dir = out_base / "figures"
    tab_dir = out_base / "tables"
    sup_dir = out_base / "supplementary"
    for d in (fig_dir, tab_dir, sup_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading per-fold data from {input_dir}/ ...")
    data = load_all(input_dir)

    # Per-topic stats
    per_topic = []
    pooled_diffs = []
    for topic in TOPICS:
        if EXPERT_MODE not in data[topic] or AUTO_MODE not in data[topic]:
            continue
        e = data[topic][EXPERT_MODE]["wss"]
        a = data[topic][AUTO_MODE]["wss"]
        diffs = [ei - ai for ei, ai in zip(e, a)]
        pooled_diffs.extend(diffs)
        boot = bootstrap_ci(diffs)
        perm_p = paired_perm_p(diffs)
        sizes = TOPIC_CHARS[topic]
        nb = nadeau_bengio(diffs, n_train=sizes["n_train"], n_test=sizes["n_test"])
        per_topic.append({
            "topic": topic,
            "n": len(diffs),
            "mean": boot["mean"],
            "ci_lo": boot["ci_lo"],
            "ci_hi": boot["ci_hi"],
            "perm_p": perm_p,
            "nb_t": nb["t_stat"],
            "nb_p": nb["p_value"],
        })

    pooled_boot = bootstrap_ci(pooled_diffs)
    pooled_perm = paired_perm_p(pooled_diffs)
    pooled = {
        "n": len(pooled_diffs),
        "mean": pooled_boot["mean"],
        "ci_lo": pooled_boot["ci_lo"],
        "ci_hi": pooled_boot["ci_hi"],
        "perm_p": pooled_perm,
    }

    print("Generating figures...")
    f1 = make_fig1(per_topic, pooled, fig_dir)
    print(f"  {f1[0]}")
    print(f"  {f1[1]}")
    f2 = make_fig2(data, fig_dir)
    print(f"  {f2[0]}")
    print(f"  {f2[1]}")

    print("Generating LaTeX tables...")
    latex_table_1(tab_dir)
    latex_table_2(tab_dir)
    latex_table_3(data, tab_dir)
    latex_table_4(per_topic, pooled, tab_dir)
    for f in sorted(tab_dir.glob("*.tex")):
        print(f"  {f}")

    print("Generating supplementary CSVs...")
    c1 = write_per_fold_csv(data, sup_dir)
    print(f"  {c1}")
    c2 = write_per_fold_diffs_csv(data, sup_dir)
    print(f"  {c2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
