# REPRODUCING.md

How to reproduce every numerical claim in the manuscript from the data in this
repository.

This document is generated from `PROVENANCE.md`, which holds one row per number
appearing in the paper, its source file, that file's date, the run conditions,
and a verification status. **If the two disagree, `PROVENANCE.md` is
authoritative.** Numbers in the paper that are pending correction are marked
here rather than silently reproduced.

Rebuilt 2026-08-05 against tree state at `6147b23`. The previous version of this
file named the wrong source for the power analysis and referenced four paths
that no longer exist; both are corrected below.

---

## Environments

**Local (`.venv312`)** — Python 3.12 on Windows. All bag-of-words runs, all
statistical analyses, figure regeneration, and the assignment analysis. Activate
from Git Bash with `source .venv312/Scripts/activate`.

**Colab T4** — BiomedBERT fine-tuning only. Both notebooks mount Google Drive
and write there; the paper-cited subset has been copied into `outputs/`, which is
the repository of record.

---

## Before you start: the benchmark cache is not distributed

`.gitignore` excludes `data/cohen/cache/`, which is the `--cache-dir` default and
the only directory the pipeline reads. A fresh clone therefore has no cached
PubMed records and the loader will re-fetch from NCBI Entrez on first run, which
requires an email address passed via `--email` and takes roughly an hour for the
three topics.

`data/cohen/pubmed_cache/` is tracked and holds the same 6,216 records, but no
code path reads it. Either copy it to `data/cohen/cache/` before running
anything, or point `--cache-dir` at it:

```bash
cp -r data/cohen/pubmed_cache data/cohen/cache
```

This is `PROVENANCE.md` #41 and is pending a proper fix.

---

## Claim-to-source map

Table numbers refer to the manuscript. Status column: **V** verified by
recomputation from the named source; **P** pending correction, see the ledger
row; **U** source identified but not independently recomputed.

### Table 1 — topic characteristics

| | |
|---|---|
| Source | `data/cohen/epc-ir.clean.tsv` plus the retrieval cache |
| Status | **P** (ledger #28) |

The TSV holds 3,465 / 1,915 / 851 rows for Statins / Opioids / ADHD. Table 1
reports 2,744 / 1,772 / 803. The difference is articles whose abstracts were not
retrievable through Entrez. Labels come from the abstract-level decision column,
mapped `I` → 1 and everything else → 0 by `parse_cohen_tsv`.

```bash
python -c "
import pandas as pd
df=pd.read_csv('data/cohen/epc-ir.clean.tsv',sep='\t',header=None,dtype=str)
for t in ['Statins','Opiods','ADHD']:
    s=df[df[0]==t]; print(t, len(s), (s[3]=='I').sum())"
```

Note: 190 Opioids and 81 ADHD rows carry numerals rather than `E`/`I` in the
decision columns and are all mapped to 0. Unresolved, ledger #31.

### Tables 2–4 — bag-of-words multi-run characterisation

| | |
|---|---|
| Raw | `outputs/bow_statins_run{1..7}.txt` (2026-06-20), `outputs/bow_opiods_run{1..7}.txt`, `outputs/bow_adhd_run{1..7}.txt` (both 2026-06-26) |
| Summary | `outputs/bow_{topic}_multirun_summary.json` |
| Regenerate summary | `python parse_bow_multirun.py outputs/bow_{topic}_run*.txt` |
| Status | **V** |

The reported column is **Reg WSS@95** (regularised), the sixth column of the
TEXT MODE COMPARISON block.

**Condition to disclose:** `bow_statins_run1.txt` is byte-identical on all
reported values to `outputs/archive/bow_statins_smoke_onednn_off.txt`, and
`run2.txt` to `..._smoke_rerun2.txt`. Run 1 was executed with
`TF_ENABLE_ONEDNN_OPTS=0`. The paper's claim that the seven reruns use identical
arguments is pending correction, ledger #21.

### Table 5 — statistical tests

| | |
|---|---|
| BoW rows | per-fold differences from `outputs/bow_{topic}_multirun_summary.json`, n=35 per topic |
| BERT rows | per-fold `wss_at_95` under `expert_runs[seed].folds[]` and `auto_runs[seed].folds[]` in `outputs/bert_{topic}_multiseed_summary.json`, n=25 per topic |
| Status | BoW **V**; BERT **P** (ledger #20) |

The BERT rows as printed are seed-level statistics (n=5) reported under an n
column reading 25 and 75. The permutation p-values 0.19, 0.31, 0.81 are 6/32,
10/32 and 26/32, which is only possible at n=5. The correct per-fold set,
recomputed independently twice and matching `outputs/bert_per_fold_bootstrap.json`:

| Topic | n | Mean | 95% CI | Perm p |
|---|---|---|---|---|
| Statins | 25 | +0.0199 | [−0.0210, +0.0621] | 0.363 |
| Opioids | 25 | −0.0481 | [−0.0977, +0.0035] | 0.083 |
| ADHD | 25 | +0.0035 | [−0.0395, +0.0423] | 0.876 |
| Pooled | 75 | −0.0083 | [−0.0357, +0.0184] | 0.549 |

```bash
python - <<'EOF'
import json, numpy as np
rng=np.random.default_rng(0); pooled=[]
for t in ["statins","opiods","adhd"]:
    d=json.load(open(f"outputs/bert_{t}_multiseed_summary.json")); diffs=[]
    for s in d["expert_runs"]:
        e=[f["wss_at_95"] for f in d["expert_runs"][s]["folds"]]
        a=[f["wss_at_95"] for f in d["auto_runs"][s]["folds"]]
        diffs+=[x-y for x,y in zip(e,a)]
    pooled+=diffs; arr=np.array(diffs)
    b=[rng.choice(arr,len(arr),replace=True).mean() for _ in range(10000)]
    print(t,len(arr),round(arr.mean(),4),np.round(np.percentile(b,[2.5,97.5]),4))
EOF
```

`bootstrap_paired_permutation.py` cannot reproduce these: it expects
`bert_{topic}_{mode}.txt` without the seed suffix. Ledger #43.

### Tables 6–8 — BiomedBERT multi-seed

| | |
|---|---|
| Raw | `outputs/bert_{topic}_{mode}_seed{42,7,13,21,31}.{txt,json}` |
| Summary | `outputs/bert_{topic}_multiseed_summary.json` (2026-06-26) |
| Regenerate | Colab T4, `notebooks/cohen_bert_multiseed.ipynb`, ~5 h for 50 trainings |
| Status | **V** |

Statins per-seed values were corrected against the JSON at commit `c7ee290`.

### Table 9 and Figure 2 — evaluation design sensitivity

| | |
|---|---|
| Source | `paper_experiments/outputs/bow_experiments_summary.csv` (2026-07-01) |
| Status | **V** for the numbers; **P** for Figure 2 (ledger #35) |

Subsampled Statins n=803: sum of `diff` over 35 rows ÷ 35 = **+0.0332**.
10-fold full corpus: sum over 70 rows ÷ 70 = **+0.0207**. Both reproduce.

Subsampling is stratified: `train_test_split(train_size=subsample_n,
random_state=subsample_seed, stratify=df["labels"])` in `src/cohen_pipeline.py`.

`make_fig2_design_sensitivity.py` hardcodes every row including the stale BERT
interval `-0.011, +0.052`. It must be edited by hand when Table 5 is corrected.
Two design-sensitivity generators exist (`make_fig2_design_sensitivity.py` and
`scripts/fig_design_sensitivity.py`) and which produced
`fig_design_sensitivity_final.pdf` is unresolved, ledger #32.

### Table 10 — empirical power analysis

| | |
|---|---|
| Source | `bow_stats_results.json` (repository root) |
| Underlying runs | `outputs/archive/bow_statins_smoke.txt`, `outputs/bow_opiods_text_modes.txt`, `outputs/bow_adhd_text_modes.txt`, all 2026-06-18 |
| Regenerate | `python paper_experiments/power_analysis.py` |
| Status | **V** for the arithmetic; **P** for the basis (ledger #22) |

MDE = (z₀.₉₇₅ + z₀.₈) × SD / √5 gives 0.0852, 0.1892, 0.2857 against the
reported 0.085, 0.189, 0.286.

**Correction to the previous version of this file**, which claimed the Statins
row came from `bow_statins_run1.txt`. It does not. `bow_stats_results.json`
records Statins per-fold diffs [0.062, 0.064, 0.210, 0.180, 0.108]; adding these
to the smoke run's auto folds reproduces that run's expert folds exactly, mean
0.235. Run 1's diffs are [0.089, 0.027, 0.155, 0.000, 0.120] and match nothing.

Table 10 therefore rests on a separate 18 June single-run session with oneDNN
enabled, two days before the Statins multi-run set and eight before the Opioids
and ADHD multi-runs. The paper describes it only as "the canonical single-run
5-fold analysis".

### Figure 1 — forest plot

| | |
|---|---|
| Generator | `scripts/make_fig1_gap_forest.py` |
| Reads | all six `bow_*_multirun_summary.json` and `bert_*_multiseed_summary.json` |
| Status | **V** for the plot; **P** for the caption (ledger #20) |

The generator computes fold differences at runtime and therefore already plots
the correct per-fold intervals. The caption quotes the stale seed-level interval,
so figure and caption currently disagree.

`scripts/make_fig1_v2.py` and `scripts/make_paper_artifacts.py` are pre-audit
baselines that assume filenames without the `_seed42` suffix. Preserved as record;
not the regeneration path.

### §3.3 — token truncation rates

| | |
|---|---|
| Source | `paper_experiments/outputs/audit_token_lengths.json` (2026-07-01) |
| Regenerate | `python paper_experiments/audit_token_lengths.py` |
| Status | **V** |

15.12 / 10.38 / 11.83 % in `title_abstract_mesh` mode. The 4.6–7.9 % band is the
true min and max across the six abstract and title_abstract values.

### §3.5 and Appendix A.3 — non-determinism

| | |
|---|---|
| BoW drift | recomputable from `outputs/bow_statins_multirun_summary.json` |
| BERT drift | `outputs/audit_comparison.json` (2026-06-24) |
| Status | **P** (ledger #24, #25) |

The paper's "drift up to 0.03 WSS@95% per fold" traces to an earlier version of
this file and is wrong. The measured maximum per-fold spread across the seven
Statins runs is **0.144** (expert mode) and **0.137** (auto mode).

```bash
python -c "
import json;d=json.load(open('outputs/bow_statins_multirun_summary.json'))
runs=[r['modes'] for r in d['runs']]
for m in ['title_abstract_mesh','auto_mesh']:
    cols=list(zip(*[r[m] for r in runs]))
    print(m, round(max(max(c)-min(c) for c in cols),4))"
```

The "0.28 WSS@95%" BERT drift attributed to Statins is ADHD's figure. Per-topic
maxima are Statins 0.2636, Opioids 0.1800, ADHD 0.2810.

### §3.2 — auto-MeSH vocabulary

| | |
|---|---|
| Source | `src/auto_mesh.py`, `build_mesh_vocabulary(cache_dir, min_length=4)` |
| Status | **P** (ledger #26) |

The vocabulary is 4,740 terms built from all 6,216 cached records. The three
topics under study total 5,319 articles, so roughly 900 records from Cohen topics
not used in the paper contribute terms. §3.2's "the topic's cached records" is
inaccurate. Matching is bare substring containment with a four-character floor
and no word boundaries.

---

## Results not yet in the manuscript

### Baseline decomposition

Each MeSH mode measured against its own no-MeSH baseline, n=35 per-fold,
10,000-resample bootstrap:

| | Mean | 95% CI |
|---|---|---|
| Expert increment (`title_abstract_mesh` − `title_abstract`) | +0.0838 | [+0.0671, +0.1001] |
| Auto increment (`auto_mesh` − `abstract`) | −0.0058 | [−0.0230, +0.0119] |
| Difference in differences | +0.0896 | [+0.0623, +0.1161] |

Source `outputs/bow_statins_multirun_summary.json`, all four modes. Resolves the
title asymmetry between the compared modes, ledger #27.

### Assignment error analysis

| | |
|---|---|
| Script | `paper_experiments/mesh_assignment_analysis.py` |
| Output | `outputs/mesh_assignment_analysis_statins.{json,txt}` |
| Run | `python -m paper_experiments.mesh_assignment_analysis --email you@example.com` |

Substring matching recovers 16.2 % of expert-assigned terms on included articles
and 15.5 % on excluded; 76–78 % of everything it matches was never assigned to
that article. Stratification by check tag, qualifier and substantive heading is
pending, ledger #33.

---

## Source modules

| Module | Purpose |
|---|---|
| `src/cohen_pipeline.py` | BoW pipeline CLI, including `--subsample-n` |
| `src/cohen_bert_pipeline.py` | BiomedBERT pipeline CLI |
| `src/benchmark_loader.py` | Entrez fetcher, TSV parser, local cache |
| `src/auto_mesh.py` | Substring-match MeSH assignment |
| `src/features.py` | **Keras `Tokenizer`**, not CountVectorizer |
| `src/evaluation.py` | WSS@95 % and screening metrics |
| `src/bert_models.py` | BiomedBERT wrapper |
| `src/preprocessing.py` | Tokenisation, stopwords, ontology enrichment |

`src/cohen_pipeline.py.bak` is the pre-subsample version, tracked in error,
ledger #49.

---

## Known defects

Sixteen repository defects and nine manuscript defects are catalogued in
`PROVENANCE.md` Parts 2 and 5, each with the correction it requires. This file
is regenerated from that ledger whenever a defect closes.
