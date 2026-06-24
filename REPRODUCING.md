# REPRODUCING.md

How to reproduce every numerical claim in the paper draft from the data
in this repository.

For each claim, this document names:

- The source data files holding the raw per-fold values.
- The summary or analysis file holding the headline number.
- The regeneration command if the summary needs to be rebuilt from the raw data.
- The environment required.

The principle: every claim in the paper is traceable from a sentence in
the manuscript back to the bits on disk that support it.

## Environments

Two environments are needed depending on which classifier was used.

**Local (`.venv312`):** Python 3.12 venv on Windows for all bag-of-words
work, the statistical analyses, and figure regeneration. TensorFlow 2.14+
is in the venv but BERT runs were not done locally. Activate with
`source .venv312/Scripts/activate` from Git Bash.

**Colab T4:** BiomedBERT fine-tuning. Both notebooks
(`notebooks/cohen_bert_audit.ipynb` and `notebooks/cohen_bert_multiseed.ipynb`)
were executed on Colab free-tier T4 GPUs. The notebooks mount Google Drive
at `/MyDrive/cohen_bert_run/` and write per-fold artifacts there. The
paper-cited subset of those artifacts has been copied into `outputs/`.

## Source of truth file map

The repository contains all paper-cited data. The Drive runtime location
(`/g/My Drive/cohen_bert_run/` on the author's Windows machine) is where
the Colab notebooks read and write. Copies in `outputs/` are the
repository-of-record.

### Bag-of-words

| Data | Repository path |
|---|---|
| Statins multi-run, seven reruns, per-fold | `outputs/bow_statins_run{1..7}.txt` |
| Statins multi-run summary (mean, range, n_runs) | `outputs/bow_statins_multirun_summary.json` |
| Opiods single run, per-mode | `outputs/bow_opiods_text_modes.txt` |
| ADHD single run, per-mode | `outputs/bow_adhd_text_modes.txt` |
| Statistical analysis (bootstrap, permutation, NB) | `outputs/bow_stats_results.json` |

### BiomedBERT audit (single seed, three topics)

| Data | Repository path |
|---|---|
| Per-fold values, twelve topic-mode combinations | `outputs/bert_{topic}_{mode}_seed42.{txt,json}` |
| Aggregated statistical analysis | `outputs/analysis_results_full_v2.json` |
| Per-fold drift between pre-audit and audit | `outputs/audit_comparison.json` |

### BiomedBERT multi-seed (Statins, five seeds)

| Data | Repository path |
|---|---|
| Per-seed per-fold values, ten topic-mode-seed combinations | `outputs/bert_statins_{title_abstract_mesh,auto_mesh}_seed{42,7,13,21,31}.{txt,json}` |
| Aggregated per-seed and pooled summary | `outputs/bert_statins_multiseed_summary.json` |

### Figures

| Figure | Repository path |
|---|---|
| Figure 1: BiomedBERT vs BoW Statins reference forest plot | `outputs/fig1_gap_forest_v2.{pdf,png}` |

## Claim-to-data map

### Claim: BoW Statins multi-run expert-vs-auto MeSH gap is +0.096 (range [+0.077, +0.114] across 7 runs)

**Source:** `outputs/bow_statins_run{1..7}.txt` (per-fold WSS@95 per text mode)
**Summary:** `outputs/bow_statins_multirun_summary.json` field `summary.gap_mean`
**Regenerate the summary from raw data:**
```bash
python scripts/parse_bow_multirun.py outputs/bow_statins_run*.txt \
  > outputs/bow_statins_multirun_summary.json
```
**Environment:** Local `.venv312`
**Caveat:** The seven reruns exist because the BoW pipeline shows residual
non-determinism between identical-command reruns (per-fold WSS drift up to
0.033). The source of non-determinism is Keras Dense layer initialisation
which the pipeline's `set_seeds()` does not fully control. Multi-run
characterisation is the methodological response.

### Claim: BoW Opiods gap is -0.010 with CI [-0.141, +0.086], ADHD gap is -0.030 with CI [-0.218, +0.136], pooled gap +0.028 with CI including zero

**Source:** `outputs/bow_opiods_text_modes.txt`, `outputs/bow_adhd_text_modes.txt`,
and the Statins Run 1 file `outputs/bow_statins_run1.txt`
**Summary:** `outputs/bow_stats_results.json` (paired bootstrap, exact
paired permutation, Nadeau-Bengio corrected t-test, per topic and pooled)
**Regenerate the statistical analysis:**
```bash
# Note: scripts/demo_statistical_analysis.py was written for BiomedBERT
# pre-audit filenames. The BoW analysis was produced by the pipeline
# code directly; the .json output is the source of truth for the paper.
```
**Environment:** Local `.venv312`
**Caveat:** Opiods and ADHD are single-run. Multi-run characterisation
analogous to the Statins work is queued before external submission.

### Claim: BiomedBERT audit per-topic gaps are Statins +0.002, Opiods -0.073, ADHD -0.055; pooled -0.042 with CI [-0.098, +0.008]

**Source:** `outputs/bert_{statins,opiods,adhd}_{abstract,title_abstract,title_abstract_mesh,auto_mesh}_seed42.{txt,json}` (12 topic-mode combinations, 5 folds each)
**Summary:** `outputs/analysis_results_full_v2.json`
**Regenerate the audit BERT runs:**
```
# Colab T4 required. Open notebooks/cohen_bert_audit.ipynb in Colab,
# mount /MyDrive/cohen_bert_run/, run all cells. Approximately 90
# minutes wall-clock for the full 12-combination single-seed run.
```
**Regenerate the statistical analysis (note caveat):**
```bash
# scripts/demo_statistical_analysis.py expects pre-audit filenames
# (bert_{topic}_{mode}.txt, no _seed42 suffix). The CU 178 §6 patch
# makes the script audit-aware. Patch pending. For now, the source
# of truth is analysis_results_full_v2.json produced by the audit
# notebook itself.
```
**Environment:** Colab T4 for the runs; local `.venv312` for analysis
**Caveat:** The audit run uses explicit `--seed 42` passed to
`src.cohen_bert_pipeline`. Per-fold drift between pre-audit (implicit RNG)
and audit (explicit seed) reaches |Δ| = 0.28 on individual folds.
`outputs/audit_comparison.json` documents the drift. The audit run is the
methodologically clean reference; the pre-audit JSON is preserved in
`outputs/archive/analysis_results_full.json`.

### Claim: BiomedBERT Statins multi-seed pooled gap is +0.020 across five seeds, per-seed range [-0.007, +0.060]

**Source:** `outputs/bert_statins_{title_abstract_mesh,auto_mesh}_seed{42,7,13,21,31}.{txt,json}` (10 topic-mode-seed combinations, 5 folds each, 25 fold values per mode)
**Summary:** `outputs/bert_statins_multiseed_summary.json`
**Regenerate the multi-seed runs:**
```
# Colab T4. Open notebooks/cohen_bert_multiseed.ipynb, mount Drive,
# run all cells. Approximately 5 hours wall-clock for 50 BERT model
# trainings (2 modes × 5 seeds × 5 folds).
```
**Environment:** Colab T4 only
**Caveat:** Statins only. Multi-seed analysis for Opiods and ADHD is
queued. Not blocking internal review.

### Claim: Figure 1 forest plot shows the BERT pooled CI does not overlap the BoW multi-run band

**Source:** `outputs/bow_statins_multirun_summary.json` (BoW band)
plus the audit BERT values from `outputs/analysis_results_full_v2.json`
**Output:** `outputs/fig1_gap_forest_v2.{pdf,png}`
**Regenerate:**
```bash
python scripts/make_fig1_v2.py \
  --input outputs/bow_statins_multirun_summary.json \
  --outdir outputs/
```
**Environment:** Local `.venv312`
**Caveat:** The BERT values in this script are hardcoded from
`Cohen_BERT_Extension_Results_Consolidation_v2.md` Section 5.2
(working draft, not in repo). The CU 178 §6 patch to
`scripts/make_paper_artifacts.py` is the canonical pipeline path that
reads from `outputs/analysis_results_full_v2.json` directly. Patch
pending.

## Pipeline scripts and their state

| Script | State | Notes |
|---|---|---|
| `scripts/parse_bow_multirun.py` | Current | Produces multi-run summary from per-fold .txt files |
| `scripts/make_fig1_v2.py` | Current | Audit-aware. Produces fig1 with multi-run BoW ribbon |
| `scripts/make_paper_artifacts.py` | Pre-audit historical baseline | Assumes filenames without `_seed42`. Builds fig1, fig2, four LaTeX tables. CU 178 §6 patch pending to make audit-aware |
| `scripts/demo_statistical_analysis.py` | Pre-audit historical baseline | Same filename assumption. Produced the original statistical analyses |

The two pre-audit scripts are preserved as historical baseline because they
produced the earlier analyses. Running them today against `outputs/`
requires either renaming the audit `_seed42` files to drop the suffix, or
applying the CU 178 §6 patch. The patch is documented but not yet executed.

## Source code

| Module | Purpose |
|---|---|
| `src/cohen_pipeline.py` | BoW pipeline CLI for the Cohen benchmark |
| `src/cohen_bert_pipeline.py` | BiomedBERT pipeline CLI for the Cohen benchmark |
| `src/benchmark_loader.py` | NCBI Entrez fetcher with local caching |
| `src/auto_mesh.py` | Substring-match MeSH assignment |
| `src/features.py` | Feature extraction (sklearn CountVectorizer) |
| `src/bert_models.py` | BiomedBERT classifier wrapping HuggingFace transformers |
| `src/evaluation.py` | WSS@95% and other screening metrics |
| `src/preprocessing.py` | Tokenisation, stopword removal, ontology enrichment |

The original thesis pipeline (`src/pipeline.py`) is preserved unchanged
from the 2026 refactor.

## Archive directories

`outputs/archive/` and `paper/archive/` contain superseded files preserved
as experimental record. Each has its own README documenting what each
file is, when it was produced, and what supersedes it. These directories
are part of the reproducibility story: how the analysis evolved is part
of the evidence base.

## Data files

The Cohen et al. (2006) benchmark data files are not in the repository.
The benchmark loader (`src/benchmark_loader.py`) fetches abstracts from
NCBI Entrez via Biopython on first run and caches them locally at
`data/cohen/cache/`. An Entrez email address is required at runtime.

The dementia corpus data files (`abstracts.tsv`, `neo.json`,
`med-stopwords.txt`) referenced by the original thesis pipeline are also
not in the repository. See `data/README.md`.
