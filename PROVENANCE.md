# Provenance Ledger

**Manuscript:** Evaluation design conditions the expert-vs-auto MeSH gap
**Built:** 2026-08-05 | **Updated:** 2026-08-11 (v7.7 merge)
**Manuscript version at update:** v7.7 NEJLT

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
5. **Locations are LaTeX labels, never printed table numbers.** Added 2026-08-11.
   Every table number in the 2026-08-05 build was v7.5-era: v7.6 inserted
   `tab:decomposition` and `tab:assignment`, shifting everything from the
   statistics table onward, and v7.7 moved three tables to Appendix C, shifting
   again. Numbers restale on every structural edit; labels do not.
6. **A verification entry states its scope.** Added 2026-08-11 after B1. "Verified"
   without a named set of rows, columns and quantities is worse than nothing,
   because it stops the next reader looking.
7. **Non-numeric claims get rows too.** Added 2026-08-11. See Part 7. Value
   provenance was checked in the 2026-08-05 audit and passed; thirteen further
   defects were found on 2026-08-11 in claim classes the audit was never designed
   to see.
8. **A path recorded as a defect is written without backticks.** Added
   2026-08-11. Backticks assert the path exists, and `verify_branch.sh` Check 3
   tests every backticked path in this file.

### Table label reference

| Label | Content |
|---|---|
| `tab:topics` | Cohen topic characteristics |
| `tab:bow_statins_multirun` | BoW Statins, seven reruns |
| `tab:bow_opiods_multirun` | BoW Opioids, seven reruns (Appendix C from v7.7) |
| `tab:bow_adhd_multirun` | BoW ADHD, seven reruns (Appendix C from v7.7) |
| `tab:decomposition` | Each augmentation against its own baseline (added v7.6) |
| `tab:assignment` | Assignment-level error analysis (added v7.6) |
| `tab:bert_statins` | BiomedBERT Statins, five seeds |
| `tab:bert_opiods` | BiomedBERT Opioids, five seeds |
| `tab:bert_adhd` | BiomedBERT ADHD, five seeds (Appendix C from v7.7) |
| `tab:stats` | Bootstrap CIs and permutation p-values |
| `tab:design_sensitivity` | Evaluation design sensitivity |
| `tab:power` | Empirical power analysis |

---

## Part 1 — Verified

| # | Value | Location | Source | Src date | Conditions | Status |
|---|---|---|---|---|---|---|
| 1 | Statins run gaps, 7 runs | `tab:bow_statins_multirun` | `outputs/bow_statins_run{1..7}.txt` | 06-20 | see #21 | VERIFIED |
| 2 | Statins multi-run mean +0.096 | `tab:bow_statins_multirun`, §4.1, abstract | as #1 | 06-20 | | VERIFIED |
| 3 | Statins run SD 0.015, range [+0.077,+0.114] | §4.1 | as #1 | 06-20 | | VERIFIED |
| 4 | Opioids run gaps, 7 runs, mean +0.007 | `tab:bow_opiods_multirun` | `outputs/bow_opiods_run{1..7}.txt` | 06-26 | no archive reuse | VERIFIED |
| 5 | ADHD run gaps, 7 runs, mean +0.006 | `tab:bow_adhd_multirun` | `outputs/bow_adhd_run{1..7}.txt` | 06-26 | no archive reuse | VERIFIED |
| 6 | BERT Statins per-seed, pooled +0.020 | `tab:bert_statins` | `outputs/bert_statins_multiseed_summary.json` | 06-26 | corrected at `c7ee290` | VERIFIED |
| 7 | BERT Opioids per-seed, pooled −0.048 | `tab:bert_opiods` | `outputs/bert_opiods_multiseed_summary.json` | 06-26 | | VERIFIED |
| 8 | BERT ADHD per-seed, pooled +0.003 | `tab:bert_adhd` | `outputs/bert_adhd_multiseed_summary.json` | 06-26 | | VERIFIED |
| 9 | BoW rows of `tab:stats`, CIs and perm p | `tab:stats` | per-fold from #1,#4,#5 | 06-20/26 | per-fold unit, n=35 | VERIFIED |
| 10 | Subsampled Statins +0.0332 | `tab:design_sensitivity`, Fig 2, abstract | `paper_experiments/outputs/bow_experiments_summary.csv` | 07-01 | 7 seeds × 5 folds | VERIFIED |
| 11 | 10-fold Statins +0.0207 | `tab:design_sensitivity`, Fig 2, abstract | as #10 | 07-01 | 7 runs × 10 folds | VERIFIED |
| 12 | **SUPERSEDED by B2.** Normal-approximation MDE 0.085 / 0.189 / 0.286 | was `tab:power` | `bow_stats_results.json` → `paper_experiments/power_analysis.py` → `power_analysis.md` | 07-01 | see #22 | SUPERSEDED |
| 13 | Single-run per-fold SDs 0.068 / 0.151 / 0.228 | `tab:power` | as #12 | 07-01 | see #22. Covers `tab:power` only, **not** `tab:design_sensitivity` — see B1 | VERIFIED, scope amended 08-11 |
| 14 | Truncation 15.12 / 10.38 / 11.83 % | §3.3, §5.1, §6, abstract | `paper_experiments/outputs/audit_token_lengths.json` | 07-01 | | VERIFIED |
| 15 | Abstract-mode truncation band 4.6–7.9 % | §3.3 | as #14 | 07-01 | true min/max of 6 values | VERIFIED |
| 16 | **Opioids** seed-13 swing 0.21 | §4.4 | `tab:bert_opiods` arithmetic: seed 21 −0.127 to seed 13 +0.082 = 0.209 | 06-26 | **Row was mislabelled "Statins" in the 08-05 build.** Corrected 08-11. The source table was right; the label was wrong | VERIFIED, label corrected |
| 17 | Non-overlap BoW/BERT Statins intervals | §4.5, Fig 1 caption | #9 vs **#19b** | | holds: +0.062 < +0.075. **The 08-05 build cited "#9 vs #24"; #24 is the drift defect and is unrelated.** Corrected 08-11 | VERIFIED, reference corrected |
| 18 | Vocabulary 4,740 terms from 6,216 records | §3.2 (moved into the paper by C10) | `data/cohen/cache` | 06-18 | see #26. **Location field read "not in paper" in the 08-05 build**; corrected 08-11 | VERIFIED, location corrected |
| 19 | Run-to-run SD of gap 0.015 | §6 | as #1 | 06-20 | gap-level, not per-fold | VERIFIED |
| 19a | Subsample code is stratified on labels, per-seed random_state | §3.6 | `src/cohen_pipeline.py` vs `.bak` diff | | confirms #10 at code level | VERIFIED |
| 19b | BERT per-fold replacement set (see #20) | `tab:stats` | recomputed from `bert_{topic}_multiseed_summary.json` `expert_runs[seed].folds[].wss_at_95` | 06-26 | matches `bert_per_fold_bootstrap.json` and Fig 1 | VERIFIED ×2 |

---

## Part 2 — Defects, with the correction each requires

| # | Defect | Location | Evidence | Fix | Status |
|---|---|---|---|---|---|
| 20 | **BERT rows of `tab:stats` are seed-level statistics under a per-fold `n` column.** Perm p 0.19/0.31/0.81 are 6/32, 10/32, 26/32, only possible at n=5. | `tab:stats` | `paper_draft_v4.tex` L293 discloses "n=15 seed-level"; disclosure removed by v7.1 | Per-fold set, recomputed twice: Statins +0.0199 [−0.021,+0.062] p 0.363; Opioids −0.0481 [−0.098,+0.004] p 0.083; ADHD +0.0035 [−0.040,+0.042] p 0.876; pooled −0.0083 [−0.036,+0.018] p 0.549 | FIXED in v7.6 (C1) |
| 21 | **Runs 1 and 2 are the archived oneDNN test runs.** | §4.1, App A.3 | timestamps 06-20 15:08, 15:28, 15:33 | Correct the identical-arguments claim; disclose the reuse | FIXED in v7.6 (C7) |
| 22 | **`tab:power` rests on a separate 18 June single-run session.** Statins expert 0.235 exceeds all seven multi-run values [0.198,0.227]. | `tab:power`, §3.6, §4.6, §5.4 | json diffs; smoke auto folds = smoke expert folds, mean 0.235 | **Partly superseded 08-11.** v7.7 retains the n=5 analysis, scopes it explicitly to the single-run design, and reports the n=35 figures alongside (B3). The remedy "recompute MDE from multi-run per-run SDs" is not what was done, and the reason is in B3 | FIXED in v7.7 by disclosure, not recomputation |
| 23 | **"the originally published value of +0.121" is not published.** | §4.1 | `outputs/archive/text_mode_comparison.txt` 04-14 | Rewrite as an earlier single-run estimate from this work | FIXED at one site in v7.6 (C8); **four further sites found 08-11**, see Part 7 P1 |
| 24 | **"drift up to 0.03 per fold" understates by ~5×.** Max per-fold spread 0.144. | §3.5 | `bow_statins_multirun_summary.json` | Replace with 0.144 | FIXED in v7.6 (C4) |
| 25 | **0.28 BERT drift attributed to Statins is ADHD's figure.** Statins 0.2636, Opioids 0.1800, ADHD 0.2810. | §3.5, App A.3 | `outputs/audit_comparison.json` 06-24 | Reassign to ADHD | FIXED in v7.6 (C5) |
| 26 | **Auto-MeSH vocabulary is benchmark-wide, not topic-scoped.** | §3.2 | `build_mesh_vocabulary` globs `cache_dir` | Describe accurately | FIXED in v7.6 (C10). **The row's stated bias direction — "inflating the gap" — was never tested and is retracted 08-11.** See Part 7 P5 |
| 27 | **The compared modes differ in two ways** (title present in mode 3, absent in mode 4). | §1, RQ1, §3.2 | `prepare_auto_mesh_texts` docstring | Report the decomposition (#29) | FIXED in v7.6 (C11, C13) |
| 28 | **`tab:topics` counts are post-retrieval with undisclosed attrition.** 3,465/1,915/851 against 2,744/1,772/803. Includes 173/48/84 against 152/43/83. | `tab:topics`, §3.1 | `epc-ir.clean.tsv` | Caption or §3.1 clause | FIXED: totals in v7.6 (C12), **inclusion-count attrition in v7.7 (D1)** |

---

## Part 3 — New results available for the manuscript

| # | Result | Source | Status |
|---|---|---|---|
| 29 | Baseline decomposition, n=35 per-fold, 10k bootstrap: expert increment **+0.0838 [+0.0671, +0.1001]**; auto increment **−0.0058 [−0.0230, +0.0119]**; difference-in-differences **+0.0896 [+0.0623, +0.1161]**; title +0.006 (5/7 runs positive) | `bow_statins_multirun_summary.json`, all four modes | VERIFIED, in `tab:decomposition` from v7.6 |
| 30 | Assignment error analysis, Statins: recovery 16.2% included / 15.5% excluded; spurious share 76–78%; 20.5 expert terms per included article, 3.3 recovered, 10.8 spurious | `outputs/mesh_assignment_analysis_statins.txt` | VERIFIED, in `tab:assignment` from v7.6. Stratification still open (#33) |

---

## Part 3b — Rows added 2026-08-11 (v7.7)

**B1. `tab:design_sensitivity` SD column.** VERIFIED, corrected.

```
Statins  n=35  mean +0.0957  sd 0.0641   (v7.6 printed 0.067)
Opiods   n=35  mean +0.0066  sd 0.1713   (v7.6 printed 0.170)
ADHD     n=35  mean +0.0059  sd 0.1620   (v7.6 printed 0.160)
```

Source: `outputs/bow_{topic}_multirun_summary.json`, accessed as
`d['runs'][i]['modes'][mode]`, per-fold `title_abstract_mesh` minus `auto_mesh`,
pooled over seven runs. Regenerated by `paper_experiments/power_analysis.py`
Part 2 as of the 08-11 patch. Means reproduce exactly; only the SDs were wrong.
`0.067`, `0.170` and `0.160` appear in no data file and no project document —
they were entered directly into the table and re-quoted in `tab:power`'s caption.

Rows 2 and 3 verify clean against
`paper_experiments/outputs/bow_experiments_summary.csv`:
subsampled `n=35 mean +0.0332 sd 0.1774`; 10-fold `n=70 mean +0.0207 sd 0.0859`.

**How this survived:** `REPRODUCING.md`'s entry for this table checks the two
means from the long CSV and marks the whole table **V**. The SD column and the
three multi-run reference rows were never checked. That entry must state what it
covers — see Rule 6.

**B2. Exact MDEs, single-run design.** VERIFIED. Supersedes #12.

```
Statins  SD 0.0675070  normal 0.0846  exact 0.1135
Opioids  SD 0.1511473  normal 0.1894  exact 0.2542
ADHD     SD 0.2281594  normal 0.2859  exact 0.3838
```

Exact noncentral-*t*, α = 0.05 two-sided, 80% power. The normal-approximation
column matches an internal session record of `power_analysis.md` to four decimals, which
confirms both the formula and the input file. `power_analysis.py` already
computed the exact ratio and reported the approximation as the headline; the
08-11 patch flips which value is printed, so `tab:power` now has a generating
script behind it rather than a hand computation. Regenerate with
`python paper_experiments/power_analysis.py`.

**B3. MDEs at the pooled fold count.** VERIFIED, newly reported in §4.6.

```
n=35, exact:  Statins 0.0313   Opioids 0.0835   ADHD 0.0790
```

All below the +0.096 multi-run Statins effect, computed under an independence
assumption §3.5 denies. **Reported in the manuscript rather than omitted**, with
the effective-sample-size question stated as unresolved. This is why #22's
original remedy was not executed: recomputing at n=35 would have replaced one
unstated scope with another.

**B4. Benchmark label counts.** VERIFIED against `epc-ir.clean.tsv`, recounted 08-11.

| Topic | Rows | Abstract `I` | Abstract `E` | Numerals | Article `I` |
|---|---|---|---|---|---|
| Statins | 3,465 | 173 | 3,291 | 1 | 85 |
| Opiods | 1,915 | 48 | 1,677 | 190 | 15 |
| ADHD | 851 | 84 | 686 | 81 | 20 |

Article-level `I` is a strict subset of abstract-level `I` in all three topics.
Post-retrieval counts are 152 / 43 / 83. In `tab:topics` and §3.1 from v7.7 (D1).

**B5. Nadeau-Bengio implementation defect.** NEW, code-level, OPEN.

`bow_stats_results.json` stores `correction_factor: 1.0` for all three topics and
`se_corrected` equal to the raw SD, so the stored *t* is mean/SD — missing the
√n and applying no correction. Correct 5-fold NB (factor 1/n + n_test/n_train = 0.45):

```
Statins  stored t 1.849 p 0.138  ->  correct t 2.756 p 0.051
Opiods   stored t -0.066 p 0.950 ->  correct t -0.099 p 0.926
ADHD     stored t -0.131 p 0.902 ->  correct t -0.195 p 0.855
```

v7.6 §4.5 cited "the corrected *t*-test on the same data" against a **BERT**
permutation *p*-value; no NB computation on BERT data exists anywhere in the
repository. v7.7 removes the claim to have applied the correction and retains
the citation as the reason naive paired *t*-tests are unsuitable (D4).
**The stored statistic is still wrong. Fix the implementation or remove the
function, so the file does not carry it.**

---

## Part 4 — Open

| # | Item | Why it matters | Check | Status |
|---|---|---|---|---|
| 31 | 190 Opioids and 81 ADHD TSV rows carry numerals, not `E`/`I` | If any encode inclusions, ~10% of Opioids labels are wrong | Compare against Cohen's codebook and published counts | **CLOSED 2026-08-11.** OHSU's codebook for `epc-ir.clean.tsv` defines `I` = included, `E` = non-specifically excluded, numerals `1`–`9` = excluded with a stated reason. Every numeral is an exclusion; `parse_cohen_tsv` mapping non-`I` to 0 is **correct**. Recount confirms 190 and 81, and adds **1 Statins numeral this row missed**. No labels wrong, no re-runs. Disclosed in §3.1 (D1) |
| 32 | Which generator produced `fig_design_sensitivity_final.pdf` | Two candidates named | Compare outputs | **CLOSED 2026-08-11.** `scripts/fig_design_sensitivity.py`. Established by `git grep savefig` across tracked `.py` on main: that script is the only one writing `fig_design_sensitivity_final.pdf`; `make_fig2_design_sensitivity.py` writes `fig2_design_sensitivity.pdf`, which nothing consumes. Figure regenerated from the patched script on 08-11 |
| 33 | Error-analysis stratification by check tag / qualifier / substantive heading | 16% denominator includes unmatchable term classes | Patch the analysis script | OPEN. §4.3 ships with the unstratified 16% and its caveat |
| 34 | Figure 1 reads the summary JSONs and plots the correct set; its **caption** quoted the stale interval | | | CLOSED, folded into #20 |
| 51 | Rule 1 backfill: numbers in the manuscript with no row | Rule 1 is violated for each | Trace or accept as unverified | OPEN. See Part 7 P8 |

---

## Part 5 — Repository defects, not manuscript

| # | Item | Status |
|---|---|---|
| 35 | Figure 2 generator hardcodes all values including the stale BERT interval `-0.011, +0.052` | **MISDIRECTED, corrected 08-11.** The row names `make_fig2_design_sensitivity.py`, which does not produce the consumed figure (#32). The shipped `fig_design_sensitivity_final.pdf` already carried `[-0.021, +0.062]`, verified by inspecting the PDF. Executing this row would have edited a file the manuscript does not include. Superseded by #52 |
| 36 | README states `features.py` is a CountVectorizer replacement; it uses `keras_text.Tokenizer` | OPEN |
| 37 | README BERT CIs match `bert_per_fold_bootstrap.json` and therefore disagree with `tab:stats` | CLOSED by #20 |
| 38 | README "What's Next" contradicts itself on multi-seed coverage | OPEN |
| 39 | README and REPRODUCING.md reference make_fig1_gap_forest_v3.py at root | OPEN. **Note: the correct path is not `scripts/make_fig1_gap_forest.py` either** — see #53. `_v3` is that script's *output*, not its name |
| 40 | Appendix A.1 lists analysis scripts at paths that do not match the current tree | Partly resolved: `scripts/bootstrap_bert_per_fold.py` restored, see #54 |
| 41 | **INVERTED.** `.gitignore` L40 excludes `data/cohen/cache/`, the `--cache-dir` default and the only directory the code reads | OPEN. Bears on the data-availability criterion and the mirror |
| 42 | internal context-update documents and consolidation drafts tracked in the repo; identity-bearing | 
| 43 | `bootstrap_paired_permutation.py` expects `bert_{topic}_{mode}.txt` without seed suffix | OPEN |
| 44 | Consolidation v4 §3 carries the same stale BERT set as `tab:stats` | OPEN. Confirmed live by a stale-value sweep 08-11, at line 35 |
| 45 | REPRODUCING.md misattributes `tab:power`'s source to `bow_statins_run1.txt`; it is `archive/bow_statins_smoke.txt` | OPEN |
| 46 | REPRODUCING.md stale paths | OPEN |
| 47 | REPRODUCING.md states Opioids/ADHD multi-run and BERT multi-seed are "queued"; both completed 06-26 | OPEN |
| 48 | REPRODUCING.md repeats the CountVectorizer error and states benchmark data is not in the repository | OPEN |
| 49 | `src/cohen_pipeline.py.bak` tracked | OPEN |
| 50 | `src/…egg-info/PKG-INFO` carries name and email but is not tracked | CLOSED |

### Rows added 2026-08-11

| # | Item | Status |
|---|---|---|
| 52 | **Figure 2 generator semantics.** `scripts/fig_design_sensitivity.py` headed its summary column "Evaluation design" and listed BiomedBERT under it; BiomedBERT at 5-fold *is* the canonical design, and what changes is the classifier. It also reported −79% for BiomedBERT as a "gap reduction vs Canonical BoW", presenting a classifier substitution as a design effect, and drew a trend line across all four points. Two further bugs found while patching: the zero-shading `xmin/xmax` divided by 3.2 on an axis 3.6 wide with no offset, so the shading was off-column; and `mpatches` was imported unused. | **FIXED 08-11**, figure regenerated and inspected |
| 53 | **`make_fig1_v2.py` is a decoy.** Not the generator behind the shipped Figure 1. Hardcodes the seed-level BERT set from Consolidation **v2** §5.2 — Statins `+0.0022 [−0.0579, +0.0817]`, Opioids `−0.0733`, ADHD `−0.0550` (**sign flip** against the correct `+0.003`), pooled `n=15`. That is the #20 defect compiled into a generator. It also sets `BOW_STATINS_PUBLISHED = 0.121`, comments it "Original published Statins gap", and **renders it as a labelled plot line reading "BoW published (+0.121)"** — the #23 defect printed as a caption. | **ARCHIVED 08-11** to `archive/`, with `make_fig2_design_sensitivity.py` |
| 53b | **The figure namespace is inverted, and that is how #53 stayed hidden.** `scripts/make_paper_artifacts.py` → `fig1_gap_forest.pdf` (**consumed**); `scripts/make_fig1_gap_forest.py` writes fig1_gap_forest_v3.pdf (not consumed); `make_fig1_v2.py` → `_v2` (not consumed, stale); `make_fig2_design_sensitivity.py` → `fig2_design_sensitivity.pdf` (not consumed, stale). The file named after the shipped figure does not produce it; the file that does is named after something else; the highest version number belongs to a file nothing consumes. | Header added to `make_fig1_gap_forest.py` 08-11; rename still OPEN |
| 54 | **`scripts/bootstrap_bert_per_fold.py` was deleted** in commit `992d227` (06 July, "move analysis scripts to root, remove scripts/ dir") — an intent that only half executed, since `scripts/` was later recreated and the file never reappeared at root. This is the script behind #20's correction; `outputs/bert_per_fold_bootstrap.json` survived. Appendix A.1's claim that the statistical analysis scripts are available was false for the BERT intervals for five weeks. | **RESTORED 08-11** from `992d227^` |
| 55 | **`paper_experiments/power_analysis.py` defects.** (a) `t_inflation_factor` bracketed the root at `[1e-9, 3*sd]`, pushing the noncentrality past scipy's stable `nct` range: it returns `None` at n=25 and n=70, works at n=5 and n=35, and the caller filtered `None` without noticing. (b) The exact ratio was computed and never applied to the reported MDE. (c) "so the values below are conservative" is emitted by this script; the manuscript's version came from here. (d) The module docstring named a third party and `find_bow_stats()` contained a Windows Google Drive path. | **FIXED 08-11.** Patched version adds a Part 2 generating B1 and B3 |
| 56 | `demo_statistical_analysis.py` carries a stale BERT value | OPEN, surfaced by the 08-11 sweep |
| 57 | `origin/v2.0-infastructure` exists and is unaudited | OPEN. Confirmed present by `git fetch --prune` on 08-11, which settles a question raised in an internal repository audit. Public surface; decide audit or delete |
| 58 | **`scripts/verify_branch.sh` Check 1 scope gap.** Its header says it covers "local path", but the `IDENTITY` pattern implements author-identity tokens plus the machine hostname only. A Windows Drive-mount path contains neither and passes. Archive identifiers are also uncovered: a DOI or preprint id carries no name but resolves to a deposit under one. | OPEN. Add a path shape and `10\.5281/zenodo\|arxiv\.org` |

---

## Part 6 — Experiment artifacts

Sourced from the session inventories that produced them. Every file below
supports a claim in the manuscript or records a decision that shaped one.

### Design-sensitivity experiments (`tab:design_sensitivity`, Figure 2)

| File | Role |
|---|---|
| `paper_experiments/run_statins_subsampling.sh` | Experiment A driver, matched corpus size |
| `paper_experiments/run_statins_10fold.sh` | Experiment B driver, 10-fold at full size |
| `paper_experiments/outputs/bow_statins_subN803_subseed{1..7}_modes.txt` | Experiment A raw output, 7 subsample seeds |
| `paper_experiments/outputs/bow_statins_kfold10_run{1..7}_modes.txt` | Experiment B raw output, 7 reruns |
| `paper_experiments/parse_bow_experiments.py` | Bootstrap CI parser and verdict generator |
| `paper_experiments/outputs/bow_experiments_summary.csv` | Long-format per-fold values, source of #10 and #11 |
| `paper_experiments/outputs/bow_experiments_summary.md` | Bootstrap CI table |
| `paper_experiments/outputs/bow_experiments_decision.txt` | Verdict text. Prompted the §5.2 rework|
| `paper_experiments/patch_cohen_pipeline.py` | Adds `--subsample-n` and `--subsample-seed` to the BoW pipeline |
| `paper_experiments/outputs/run_statins_{10fold,subsampling}_20260630_*.log` ×8 | **Run artifacts for these two experiments.** Referenced here so they are not orphans; they are the execution record behind #10 and #11 |

### Multi-run characterisation (`tab:bow_*_multirun`)

| File | Role |
|---|---|
| `scripts/run_bow_multirun.sh` | **Driver that produced the seven reruns behind #1, #4, #5 and therefore every BoW table.** Referenced here so it is not an orphan |
| `outputs/bow_{topic}_multirun_summary.json` | Per-run, per-fold values. Source of B1 |

### Power analysis and token audit (`tab:power`, §3.3)

| File | Role |
|---|---|
| `paper_experiments/power_analysis.py` | MDE table generator. Patched 08-11, see #55 |
| `paper_experiments/outputs/power_analysis.md` | MDE table, source of #13 and B2, and of B1 and B3 from the Part 2 addition |
| `paper_experiments/audit_token_lengths.py` | Truncation-rate analysis |
| `paper_experiments/outputs/audit_token_lengths.md` | Truncation rates, source of #14 and #15 |
| `scripts/bootstrap_bert_per_fold.py` | **Produces `outputs/bert_per_fold_bootstrap.json`, the source of #19b and the #20 correction.** Restored 08-11, see #54 |

### Audit and reviewer-facing analyses

| File | Role |
|---|---|
| `paper_experiments/audit_bow_bert_data_parity.md` | Narrative audit of BoW/BERT data parity |
| `outputs/analysis_results_full_v2.json` | Audit-based three-topic single-seed analysis. Verification source for `tab:bert_statins` per-fold values |
| `outputs/audit_comparison.json` | Pre- and post-audit drift, source of #25 |
| `paper_experiments/local_inspect.sh` | One-shot diagnostic, retained for reproducibility |
| `scripts/build_manifest.py` | Generates `MANIFEST.md` |
| `scripts/verify_branch.sh` | Five-check pre-mirror verification. Scope gap at #58 |
| `scripts/verify_v7_7.sh` | Ten-check verification that the tree matches manuscript v7.7 |
| `scripts/apply_v7_7_repo.sh` | Applies the mechanical v7.7 repository changes; generates `docs/CORRECTIONS_v7_7.md` |
| `scripts/step1_docs.sh` | Documentation-cleanup worklist reporter and pre-commit gate |
| `docs/CORRECTIONS_v7_7.md` | Generated worklist of documentation defects |
| `MANIFEST.md` | Generated inventory |
| `notebooks/cohen_bert_audit.ipynb` | Colab notebook for the BiomedBERT reproducibility audit |
| `paper_experiments/README_paper_experiments.md` | Directory guide |
| `docs/Cohen_BERT_Extension_Results_Consolidation_v4.md` | Consolidated results record. **Carries three defects listed in the addendum section D and must be corrected with the manuscript** |

### Figures

**Corrected 2026-08-11.** The 08-05 build named the wrong generator for **both**
figures. Established by `git grep savefig` across tracked `.py` on main.

| File | Generator | 08-05 build said |
|---|---|---|
| `outputs/fig1_gap_forest.pdf`, `.png` | `scripts/make_paper_artifacts.py` | `scripts/make_fig1_gap_forest.py` — **wrong**, that script writes `fig1_gap_forest_v3.pdf` |
| `outputs/fig_design_sensitivity_final.pdf`, `.png` | `scripts/fig_design_sensitivity.py` | `make_fig2_design_sensitivity.py` — **wrong**, that script writes `fig2_design_sensitivity.pdf`, which nothing consumes |

The 08-05 build therefore asserted in Part 6 what Part 4 #32 simultaneously
recorded as open, and asserted it wrongly. See #32 and #53b.

### Superseded, retained as record

| File | Role |
|---|---|
| `outputs/archive/bow_statins_smoke.txt` | Source of `tab:power`, see #22 |
| `outputs/archive/bow_statins_smoke_onednn_off.txt`, `bow_statins_smoke_rerun2.txt` | oneDNN falsification runs, see #21 |
| `outputs/archive/text_mode_comparison.txt` | 14 April run, the +0.121 of #23 |
| `outputs/archive/all_workflows_statins.txt` | April exploration |
| `outputs/archive/bert_val_tuned.txt` | Early tuned BERT validation log |
| `outputs/archive/analysis_results_full.json` | Pre-audit single-seed analysis |
| `archive/make_fig1_v2.py` | See #53 |
| `archive/make_fig2_design_sensitivity.py` | See #35, #53b |
| `archive/pre_v7_7/` | Pre-patch copies of every file `scripts/apply_v7_7_repo.sh` replaced |
| `archive/fig1_gap_forest_v2.pdf` | Output of the decoy generator, see #53 |
| `archive/fig1_gap_forest_v2.png` | Output of the decoy generator, see #53 |
| `archive/commit_and_store.sh` | Superseded session helper |
| `archive/verify_setup.sh` | Superseded setup check. `paper_experiments/README_paper_experiments.md` still cites the pre-archive path; update it or restore the file |

---

## Part 7 — Claim register

**Added 2026-08-11.** The 2026-08-05 audit verified that every number traced to a
source file. It was correct and it caught real defects. It missed thirteen more,
because numbers were never the exposure. This part records assertions that are
not numbers: literature attributions, magnitude relations between verified
numbers, directional claims, statements about statistical validity, and figure
semantics.

| # | Claim | Evidence | Status |
|---|---|---|---|
| P1 | "reproduces the effect reported by prior work" and variants | The ~0.10 traces to this work's own 14 April and 18 June runs (#23), not to any published value. **#23 was fixed at §4.1 in v7.6 and survived at four further sites**: Introduction H1, §5.1, §5.3, Conclusion. Each contradicted the Introduction's own novelty claim | FIXED in v7.7 (D9). **Lesson: a change-log entry that fixes a location does not fix a claim** |
| P2 | "an order of magnitude below the canonical reference" | 0.096/0.021 = 4.6 and 0.096/0.020 = 4.8. Figure 2's own panel prints −78% and −79%, so the text contradicted its own figure | FIXED in v7.7 (D7). Origin: Consolidation v4 §3 claim 3 |
| P3 | "the bootstrap and permutation procedures are conservative under that dependence" | For a mean of positively correlated observations, true Var = (σ²/n)(1+(n−1)ρ) > σ²/n, so the iid bootstrap **understates** the variance. The interval is narrower than the dependence warrants, not wider. The sentence asserted the opposite, framed as reassurance | FIXED in v7.7 (D4) |
| P4 | Interval overlap and non-overlap used as a test, at five sites, in both directions, while §3.5 stated intervals are not decision procedures | §4.5 non-overlap; §4.6 overlap for similarity; §4.6 "within sampling noise"; §5.1 non-overlap for difference; Figure 1 caption | FIXED in v7.7 (D10) |
| P5 | "a larger vocabulary … works against mode 4"; "the reported gap is if anything an upper bound" | Asserted, never tested. §4.3's spurious-match share is descriptive and says nothing about the effect on WSS@95%. **#26 records the same untested direction as a finding** | FIXED in v7.7 (D2); #26 corrected above |
| P6 | "Truncation therefore biases the observed gap toward zero" | Direction not established; contradicted the abstract's own hedge on the same point | FIXED in v7.7 (D11) |
| P7 | "decomposing the canonical-design magnitude into distinct components" | The two perturbations are one-factor changes from a common baseline; they do not compose, the joint design was never run, and §6 called the same decomposition approximate | FIXED in v7.7 (D12) |
| P8 | **Rule 1 backfill.** Numbers in the manuscript with no row: Cohen 2006 cross-topic average 18.5% and "four topics below 5%" (§2); pilot AUC ≈ 0.43 (§3.3); per-fold WSS SD ≈ 0.07 (§4.1); fold-positive counts 32/35, 23/35, 18/35 (§4.1, derivable from the BoW tables); four-seed Opioids mean −0.081 (§4.4); term frequencies 148, 97, 58, 39, 53 (§4.3, #30 covers the aggregate only) | None is known to be wrong; all are unverified | OPEN, tracked as #51 |
| P9 | **Scope claims.** A number computed at one design and quoted about another. `tab:power` is computed at n=5; the abstract described its MDEs as characterising "the present per-topic fold counts", which are 35 | The hardest defect class to see, because both the number and its source are correct | FIXED in v7.7 (D5). Rule: every reported statistic carries its design in the sentence that reports it |
| P10 | **Figure semantics.** Axis labels, column headers, connecting lines and groupings are claims a caption cannot retract | See #52 | FIXED in v7.7 (D20) and #52 |

---

## Changelog

| Date | Change |
|---|---|
| 2026-08-05 | Built from full-session audit at HEAD `6147b23` |
| 2026-08-05 | `tab:stats` replacement set verified by independent recomputation (#19b). #22 confirmed by arithmetic. #29 gains bootstrap CIs. #34 closed. #41 inverted. #50 closed. #45–#49 added. ~~Audit closed; no blocking unknowns remain.~~ **Retracted 2026-08-11: that sentence was written while #31, #32 and #33 stood OPEN in the same document, and it is the reason #31 was carried into the manuscript's "tracked, not done here" list rather than treated as blocking.** |
| 2026-08-11 | v7.7 merge. Rules 5, 6 and 7 added. All locations converted from printed table numbers to labels (every number in the 08-05 build was v7.5-era). #31 and #32 CLOSED on evidence. #35 recorded as misdirected. #12 superseded by B2; #13, #16, #17, #18, #22, #26 corrected. B1–B5 added. #52–#58 added. Part 6 figure generators corrected — the 08-05 build named the wrong script for **both** figures. Part 7 claim register created from the thirteen defects the value-provenance audit was not designed to see. |
