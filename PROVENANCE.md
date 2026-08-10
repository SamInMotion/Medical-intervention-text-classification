# Provenance Ledger

**Manuscript:** Evaluation design conditions the expert-vs-auto MeSH gap
**Built:** 2026-08-05 | **Updated:** 2026-08-05 (post-audit close)
**Repo HEAD:** `6147b23` (tag `v1.1-arxiv`)
**Manuscript version at build:** v7.5 NEJLT

## Purpose

Every number that appears in the manuscript, mapped to the file that produced it,
that file's date, the run conditions where they differ, and a verification status.

Built because the paper's numbers lived in the paper and their provenance lived
nowhere, so each session re-derived the same lineage from scratch.

**Rules of use**

1. No number enters the manuscript without a row here.
2. Any manuscript edit that changes a value updates the row in the same session.
3. Status `VERIFIED` means recomputed from the named source, not inherited from a
   neighbouring claim. Verifying one row does not clear adjacent rows.
4. Open items stay as rows with status `OPEN`, never as absences.

---

## Part 1 — Verified

| # | Value | Location | Source | Src date | Conditions | Status |
|---|---|---|---|---|---|---|
| 1 | Statins run gaps, 7 runs | Table 2 | `outputs/bow_statins_run{1..7}.txt` | 06-20 | see #21 | VERIFIED |
| 2 | Statins multi-run mean +0.096 | Table 2, §4.1, abstract | as #1 | 06-20 | | VERIFIED |
| 3 | Statins run SD 0.015, range [+0.077,+0.114] | §4.1 | as #1 | 06-20 | | VERIFIED |
| 4 | Opioids run gaps, 7 runs, mean +0.007 | Table 3, §4.1 | `outputs/bow_opiods_run{1..7}.txt` | 06-26 | no archive reuse | VERIFIED |
| 5 | ADHD run gaps, 7 runs, mean +0.006 | Table 4, §4.1 | `outputs/bow_adhd_run{1..7}.txt` | 06-26 | no archive reuse | VERIFIED |
| 6 | BERT Statins per-seed, pooled +0.020 | Table 6 | `outputs/bert_statins_multiseed_summary.json` | 06-26 | corrected at `c7ee290` | VERIFIED |
| 7 | BERT Opioids per-seed, pooled −0.048 | Table 7 | `outputs/bert_opiods_multiseed_summary.json` | 06-26 | | VERIFIED |
| 8 | BERT ADHD per-seed, pooled +0.003 | Table 8 | `outputs/bert_adhd_multiseed_summary.json` | 06-26 | | VERIFIED |
| 9 | BoW rows of Table 5, CIs and perm p | Table 5 | per-fold from #1,#4,#5 | 06-20/26 | per-fold unit, n=35 | VERIFIED |
| 10 | Subsampled Statins +0.0332 | Table 9, Fig 2, abstract | `paper_experiments/outputs/bow_experiments_summary.csv` | 07-01 | 7 seeds × 5 folds | VERIFIED |
| 11 | 10-fold Statins +0.0207 | Table 9, Fig 2, abstract | as #10 | 07-01 | 7 runs × 10 folds | VERIFIED |
| 12 | MDE 0.085 / 0.189 / 0.286 | Table 10 | `bow_stats_results.json` | 07-01 | see #22 | VERIFIED |
| 13 | Table 10 per-fold SDs 0.068/0.151/0.228 | Table 10 | as #12 | 07-01 | see #22 | VERIFIED |
| 14 | Truncation 15.12 / 10.38 / 11.83 % | §3.3, §5.1, §6, abstract | `paper_experiments/outputs/audit_token_lengths.json` | 07-01 | | VERIFIED |
| 15 | Abstract-mode truncation band 4.6–7.9 % | §3.3 | as #14 | 07-01 | true min/max of 6 values | VERIFIED |
| 16 | Statins seed-13 swing 0.21 | §4.2 | Table 7 arithmetic | 06-26 | | VERIFIED |
| 17 | Non-overlap BoW/BERT Statins | §4.5, Fig 1 caption | #9 vs #24 | | holds: +0.062 < +0.075 | VERIFIED |
| 18 | Vocabulary 4,740 terms | not in paper | `data/cohen/cache`, 6,216 records | 06-18 | see #26 | VERIFIED |
| 19 | Run-to-run SD of gap 0.015 | §6 | as #1 | 06-20 | gap-level, not per-fold | VERIFIED |
| 19a | Subsample code is stratified on labels, per-seed random_state | §3.6 | `src/cohen_pipeline.py` vs `.bak` diff | | confirms #10 at code level | VERIFIED |
| 19b | BERT per-fold replacement set (see #20) | Table 5 | recomputed from `bert_{topic}_multiseed_summary.json` `expert_runs[seed].folds[].wss_at_95` | 06-26 | matches `bert_per_fold_bootstrap.json` and Fig 1 | VERIFIED ×2 |

---

## Part 2 — Defects, with the correction each requires

| # | Defect | Location | Evidence | Fix |
|---|---|---|---|---|
| 20 | **BERT rows of Table 5 are seed-level statistics under a per-fold `n` column.** Perm p values 0.19/0.31/0.81 are 6/32, 10/32, 26/32, only possible at n=5. Bootstrap of the 5 per-seed gaps reproduces the means and the v4 parenthetical [−0.037,+0.020] exactly. | Table 5 | `paper_draft_v4.tex` L293 discloses "n=15 seed-level"; disclosure removed by v7.1 | Replace with the per-fold set, independently recomputed twice: Statins +0.0199 [−0.021,+0.062] p 0.363; Opioids −0.0481 [−0.098,+0.004] p 0.083; ADHD +0.0035 [−0.040,+0.042] p 0.876; pooled −0.0083 [−0.036,+0.018] p 0.549. Figure 1 already plots this set; Figure 2 does not (see #35) |
| 21 | **Runs 1 and 2 are the archived oneDNN test runs.** All 24 values in `run1.txt` match `archive/bow_statins_smoke_onednn_off.txt`; `run2.txt` matches `bow_statins_smoke_rerun2.txt`. Run 1 executed with `TF_ENABLE_ONEDNN_OPTS=0`. | §4.1 "identical arguments"; App A.3 | timestamps 06-20 15:08, 15:28, 15:33 | Correct the identical-arguments claim; disclose the reuse |
| 22 | **Table 10 rests on a separate 18 June single-run session** with oneDNN enabled, two days before the multi-run set and eight before Opioids/ADHD multi-runs. Statins expert 0.235 exceeds all seven multi-run values [0.198,0.227]. | Table 10, §3.6, §4.4, §5.4 | Confirmed: json diffs + smoke auto folds = smoke expert folds, mean 0.235. Run 1 does not match. REPRODUCING.md claims run 1 (see #45) | Disclose the separate session and environment, or recompute MDE from multi-run per-run SDs |
| 23 | **"the originally published value of +0.121" is not published.** It is the 14 April run, expert 0.223 − auto 0.102. Contradicts the Introduction's no-prior-study claim. | §4.1 | `outputs/archive/text_mode_comparison.txt` 04-14; `PUBLISHED_STATINS_GAP` in deleted code | Rewrite as an earlier single-run estimate from this work |
| 24 | **"drift up to 0.03 WSS@95% per fold" understates by ~5×.** Max per-fold spread across 7 runs: 0.144 expert, 0.137 auto. No source for 0.03 anywhere in the repo. | §3.5 | computed from `bow_statins_multirun_summary.json` | Replace with 0.144; strengthens the multi-run justification |
| 25 | **0.28 BERT drift attributed to Statins is ADHD's figure.** Statins 0.2636, Opioids 0.1800, ADHD 0.2810. | §3.5, App A.3 | `outputs/audit_comparison.json` 06-24 | Reassign to ADHD or restate as cross-topic max |
| 26 | **Auto-MeSH vocabulary is benchmark-wide, not topic-scoped.** 4,740 terms from 6,216 cached records; the three topics total 5,319 articles. Larger vocabulary means more spurious matches in auto mode, inflating the gap. | §3.2 | `build_mesh_vocabulary` globs `cache_dir` | Describe accurately; note bias direction in §6 |
| 27 | **The compared modes differ in two ways.** `auto_mesh` = abstract + auto terms; `title_abstract_mesh` = title + abstract + expert terms. Contradicts "varying only the assignment mechanism". | §1, RQ1, §3.2 | `prepare_auto_mesh_texts` docstring | Report the decomposition (#29); confound falls out |
| 28 | **Table 1 counts are post-retrieval with undisclosed attrition.** TSV 3,465 / 1,915 / 851 against 2,744 / 1,772 / 803. Includes 173 / 48 / 84 against 152 / 43 / 83. | Table 1, §3.1 | `epc-ir.clean.tsv` | Caption or §3.1 clause stating retrieval attrition |

---

## Part 3 — New results available for the manuscript

| # | Result | Source | Status |
|---|---|---|---|
| 29 | Baseline decomposition, n=35 per-fold, 10k bootstrap: expert increment **+0.0838 [+0.0671, +0.1001]**; auto increment **−0.0058 [−0.0230, +0.0119]**; difference-in-differences **+0.0896 [+0.0623, +0.1161]**; title contribution +0.006 (5/7 runs positive) | `bow_statins_multirun_summary.json`, all four modes | VERIFIED |
| 30 | Assignment error analysis, Statins: recovery 16.2% included / 15.5% excluded; spurious share 76–78%; 20.5 expert terms per included article, 3.3 recovered, 10.8 spurious | `outputs/mesh_assignment_analysis_statins.txt` | VERIFIED, stratification pending |

---

## Part 4 — Open

| # | Item | Why it matters | Check |
|---|---|---|---|
| 31 | 190 Opioids and 81 ADHD TSV rows carry numerals, not `E`/`I`, in the decision columns. `parse_cohen_tsv` maps all non-`I` to 0. | If any encode inclusions, ~10% of Opioids labels are wrong | Inspect raw rows; compare against Cohen's published counts |
| 32 | Which generator produced `fig_design_sensitivity_final.pdf` | Two candidates: `make_fig2_design_sensitivity.py`, `scripts/fig_design_sensitivity.py` | Compare outputs |
| 33 | Error-analysis stratification by check tag / qualifier / substantive heading | 16% denominator includes unmatchable term classes | Patch the analysis script |
| 34 | CLOSED. Figure 1 reads the summary JSONs and computes fold diffs at runtime, so it plots the correct set. Its **caption** quotes the stale interval — fold into #20. | | CLOSED |

---

## Part 5 — Repository defects, not manuscript

| # | Item |
|---|---|
| 35 | Figure 2 generator hardcodes all values including the stale BERT interval `-0.011, +0.052` |
| 36 | README states `features.py` is a CountVectorizer replacement; it uses `keras_text.Tokenizer`. The paper is right, the README is wrong |
| 37 | README BERT CIs match `bert_per_fold_bootstrap.json` and therefore disagree with Table 5. README is correct; resolved by #20 |
| 38 | README "What's Next" claims multi-seed covers Statins only, then all three, in consecutive sentences |
| 39 | README and REPRODUCING.md reference make_fig1_gap_forest_v3.py at root; actual path `scripts/make_fig1_gap_forest.py` after `eac66cf` |
| 40 | Appendix A.1 lists analysis scripts at paths that do not match the current tree |
| 41 | **INVERTED.** `.gitignore` L40 excludes `data/cohen/cache/`, which is the `--cache-dir` default and the only directory the code reads. `data/cohen/pubmed_cache/` is the tracked copy and is referenced by nothing. A clone gets 6,216 records at a dead path and re-fetches from Entrez. Bears on the data-availability criterion and on the anonymised mirror. |
| 42 | `docs/Context_Update_188.md` and consolidation drafts tracked in the repo; identity-bearing |
| 43 | `bootstrap_paired_permutation.py` expects `bert_{topic}_{mode}.txt` without seed suffix; cannot reproduce Table 5 |
| 44 | Consolidation v4 §3 carries the same stale BERT set as Table 5 |
| 45 | **REPRODUCING.md misattributes Table 10's source** to `bow_statins_run1.txt`; arithmetic proves it is `archive/bow_statins_smoke.txt`. The doc is wrong about its own repository, not merely stale. |
| 46 | REPRODUCING.md stale paths: `outputs/bow_stats_results.json` (actual: root), `scripts/parse_bow_multirun.py` (actual: root), `scripts/make_fig1_v2.py` and `fig1_gap_forest_v2` (superseded at `eac66cf` by `scripts/make_fig1_gap_forest.py`) |
| 47 | REPRODUCING.md states Opioids/ADHD multi-run and BERT multi-seed are "queued"; both completed 06-26 |
| 48 | REPRODUCING.md repeats the CountVectorizer error (see #36) and states benchmark data is not in the repository (see #41) |
| 49 | `src/cohen_pipeline.py.bak` tracked; it is the pre-subsample version of the live pipeline |
| 50 | CLOSED. `src/…egg-info/PKG-INFO` carries name and email but is **not** tracked, so it does not reach the mirror. |

---

## Changelog

| Date | Change |
|---|---|
| 2026-08-05 | Built from full-session audit at HEAD `6147b23` |
| 2026-08-05 | Table 5 replacement set verified by independent recomputation (#19b). #22 confirmed by arithmetic. #29 gains bootstrap CIs. #34 closed. #41 inverted. #50 closed. #45–#49 added. Audit closed; no blocking unknowns remain. |

---

## Part 6 — Experiment artifacts

Sourced from the session inventories that produced them. Every file below
supports a claim in the manuscript or records a decision that shaped one.

### Design-sensitivity experiments (Table 9, Figure 2)

| File | Role |
|---|---|
| `paper_experiments/run_statins_subsampling.sh` | Experiment A driver, matched corpus size |
| `paper_experiments/run_statins_10fold.sh` | Experiment B driver, 10-fold at full size |
| `paper_experiments/outputs/bow_statins_subN803_subseed{1..7}_modes.txt` | Experiment A raw output, 7 subsample seeds |
| `paper_experiments/outputs/bow_statins_kfold10_run{1..7}_modes.txt` | Experiment B raw output, 7 reruns |
| `paper_experiments/parse_bow_experiments.py` | Bootstrap CI parser and verdict generator |
| `paper_experiments/outputs/bow_experiments_summary.csv` | Long-format per-fold values, source of #10 and #11 |
| `paper_experiments/outputs/bow_experiments_summary.md` | Bootstrap CI table |
| `paper_experiments/outputs/bow_experiments_decision.txt` | Verdict text. Prompted the §5.2 rework (CU 210) |
| `paper_experiments/patch_cohen_pipeline.py` | Adds `--subsample-n` and `--subsample-seed` to the BoW pipeline |

### Power analysis and token audit (Table 11, §3.3)

| File | Role |
|---|---|
| `paper_experiments/power_analysis.py` | MDE table generator |
| `paper_experiments/outputs/power_analysis.md` | MDE table, source of #12 and #13 |
| `paper_experiments/audit_token_lengths.py` | Truncation-rate analysis |
| `paper_experiments/outputs/audit_token_lengths.md` | Truncation rates, source of #14 and #15 |

### Audit and reviewer-facing analyses

| File | Role |
|---|---|
| `paper_experiments/audit_bow_bert_data_parity.md` | Narrative audit of BoW/BERT data parity |
| `outputs/analysis_results_full_v2.json` | Audit-based three-topic single-seed analysis. Verification source for Table 6 per-fold values |
| `outputs/audit_comparison.json` | Pre- and post-audit drift, source of #25 |
| `paper_experiments/local_inspect.sh` | One-shot diagnostic, retained for reproducibility |
| `scripts/build_manifest.py` | Generates `MANIFEST.md` by deriving each file's justification from imports and documentation references |
| `scripts/verify_branch.sh` | Five-check pre-mirror verification: identity, provenance fingerprint, documentation paths, imports and tests, orphans |
| `MANIFEST.md` | Generated inventory: every tracked file with the reason it is present |
| manuscript | Not in the repository. The live source is in Overleaf; the arXiv v2 source is preserved at tag `v1.1-arxiv` (`6147b23`), and the pre-audit tree is archived at `../archive_pre_anon_v1.1-arxiv.zip`. `paper/` was removed from `main` at `026e488` |
| `notebooks/cohen_bert_audit.ipynb` | Colab notebook for the BiomedBERT reproducibility audit; produced `outputs/audit_comparison.json` |
| `paper_experiments/README_paper_experiments.md` | Directory guide to the design-sensitivity and power-analysis scripts |

### Figures

| File | Role |
|---|---|
| `outputs/fig1_gap_forest.pdf`, `.png` | Figure 1, from `scripts/make_fig1_gap_forest.py` |
| `outputs/fig_design_sensitivity_final.pdf`, `.png` | Figure 2, from `scripts/fig_design_sensitivity.py`. Values are hardcoded at `ci_lower`/`ci_upper` and in the summary-table rows; both must be updated by hand when Table 6 changes |
| `make_fig2_design_sensitivity.py` | Superseded. Writes `fig2_design_sensitivity.pdf`, which nothing consumes |

### Superseded, retained as record

| File | Role |
|---|---|
| `outputs/archive/bow_statins_smoke.txt` | Source of Table 11, see #22 |
| `outputs/archive/bow_statins_smoke_onednn_off.txt`, `bow_statins_smoke_rerun2.txt` | oneDNN falsification runs, see #21 |
| `outputs/archive/text_mode_comparison.txt` | 14 April run, the +0.121 of #23 |
| `outputs/archive/all_workflows_statins.txt` | April exploration, all 11 workflows on Statins |
| `outputs/archive/bert_val_tuned.txt` | Early tuned BERT validation log |
| `outputs/archive/analysis_results_full.json` | Pre-audit single-seed analysis |
