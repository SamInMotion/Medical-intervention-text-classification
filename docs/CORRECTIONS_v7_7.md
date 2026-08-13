# Documentation corrections owed by v7.7

Generated 2026-08-11 13:13 by scripts/apply_v7_7_repo.sh.
Detected, not fixed. Each entry is file:line and the text as it stands.

## stale BERT interval / seed-level values

Superseded by the per-fold set (ledger #20). PROVENANCE.md rows that record the correction are legitimate; anything else is a live carrier.

```
PROVENANCE.md:94:| 35 | Figure 2 generator hardcodes all values including the stale BERT interval `-0.011, +0.052` |
REPRODUCING.md:161:interval `-0.011, +0.052`. It must be edited by hand when Table 5 is corrected.
docs/Cohen_BERT_Extension_Results_Consolidation_v4.md:35:| BERT (5-seed multi-seed) | Statins | +0.020 | [−0.011, +0.052] | 25 |
```

## 'order of magnitude'

The factor is about five, not ten. Originates in Consolidation v4 §3 claim 3.

```
docs/Cohen_BERT_Extension_Results_Consolidation_v4.md:66:3. **BiomedBERT under canonical evaluation produces a Statins gap comparable to BoW under 10-fold evaluation.** BERT 5-fold: +0.020. BoW 10-fold: +0.021. Within sampling noise of each other. Both an order of magnitude below BoW 5-fold reference. This reframes the "absorption" claim of v3.
```

## t-correction understated as under 4 percent

It is +33% at n=5. The 3.0% figure is the value at n=35, so the check that licensed the normal approximation was run at the wrong fold count.

```
docs/Cohen_BERT_Extension_Results_Consolidation_v4.md:119:Computed from the canonical 5-fold per-fold variance distributions, available as `bow_stats_results.json["{topic}"]["diffs"]`. MDE formula: `(z_{1-α/2} + z_{power}) × SD / sqrt(n)`. With α=0.05 two-sided, power=0.80, n=5 folds, normal approximation. T-correction shifts MDE by < 4% at n=5.
```

## effect attributed to the literature

Traces to this work's own April and June runs (ledger #23), not to any published value.

```
docs/Cohen_BERT_Extension_Results_Consolidation_v4.md:80:The 5-fold full-n result is the point estimate most cleanly recovered from the Cohen benchmark but is not the typical operational value. The CIs of canonical and 10-fold do not overlap, so this is a real difference between designs rather than sampling noise. Under this reading, the literature's reported ~0.10 WSS@95% expert-MeSH advantage on Statins is an overestimate of the typical effect.
```

## superseded MDE values

Replaced by the exact noncentral-t values 0.114 / 0.254 / 0.384. Check each hit: some are legitimate audit records.

```
PROVENANCE.md:41:| 12 | MDE 0.085 / 0.189 / 0.286 | Table 10 | `bow_stats_results.json` | 07-01 | see #22 | VERIFIED |
REPRODUCING.md:175:MDE = (z₀.₉₇₅ + z₀.₈) × SD / √5 gives 0.0852, 0.1892, 0.2857 against the
REPRODUCING.md:176:reported 0.085, 0.189, 0.286.
docs/Cohen_BERT_Extension_Results_Consolidation_v4.md:52:| Statins | +0.125 | 0.068 | 0.085 (detectable) |
docs/Cohen_BERT_Extension_Results_Consolidation_v4.md:53:| Opioids | −0.010 | 0.151 | 0.189 (1.5× Statins effect) |
docs/Cohen_BERT_Extension_Results_Consolidation_v4.md:54:| ADHD | −0.030 | 0.228 | 0.286 (2.3× Statins effect) |
paper_experiments/outputs/bow_experiments_summary.md:9:| Statins full (n≈2,744), 10-fold (NEW) | 70 | 7 | +0.0207 | [+0.0007, +0.0405] | 0.0859 |
paper_experiments/outputs/power_analysis.md:12:| Opiods | 1772 | 5 | -0.0100 | [-0.1414, +0.0860] | 0.1511 | 0.0676 | 0.2542 | 0.1894 |
```

## old Table 9 SD column

Corrected to 0.064 / 0.171 / 0.162 from outputs/bow_*_multirun_summary.json.

_none found_

## 'design-limited' claim

Scoped in v7.7 to the single-run design. Context updates carrying the unscoped claim need the same scoping.

```
docs/Cohen_BERT_Extension_Results_Consolidation_v4.md:19:3. **Revised conclusions (§7).** The "topic-stratified absorption" reading of v3 is restated as design-conditional. The cross-topic null at Opioids/ADHD is reclassified as design-limited rather than informative.
docs/Cohen_BERT_Extension_Results_Consolidation_v4.md:68:4. **The Opioids/ADHD nulls are design-limited rather than informative.** Empirical MDE at those topics' variances exceeds the Statins effect size. A Statins-sized effect at those topics would not have been detected with the present sample sizes. We cannot conclude "no effect there" — only "no effect detectable under the present design."
paper_experiments/commit_and_store.sh:118:- Q2: power analysis shows Opioids/ADHD nulls are design-limited
```

## superseded X1 instruction

X1 told you to edit and regenerate this file. The consumed figure comes from scripts/fig_design_sensitivity.py; the named file is superseded and now archived.

```
MANIFEST.md:54:| `make_fig2_design_sensitivity.py` | entry point; cited in PROVENANCE.md, REPRODUCING.md |
PROVENANCE.md:84:| 32 | Which generator produced `fig_design_sensitivity_final.pdf` | Two candidates: `make_fig2_design_sensitivity.py`, `scripts/fig_design_sensitivity.py` | Compare outputs |
PROVENANCE.md:171:| `make_fig2_design_sensitivity.py` | Superseded. Writes `fig2_design_sensitivity.pdf`, which nothing consumes |
REPRODUCING.md:160:`make_fig2_design_sensitivity.py` hardcodes every row including the stale BERT
REPRODUCING.md:162:Two design-sensitivity generators exist (`make_fig2_design_sensitivity.py` and
scripts/README.md:12:| `make_fig2_design_sensitivity.py` | **Historical** | Generated Figure 2 (design sensitivity) in earlier versions. Retained for provenance. |
scripts/fig_design_sensitivity.py:31:      not the seed-level set that make_fig2_design_sensitivity.py still holds.
```

## 'Audit closed; no blocking unknowns remain'

Written over three OPEN rows. Correct the changelog line.

```
PROVENANCE.md:118:| 2026-08-05 | Table 5 replacement set verified by independent recomputation (#19b). #22 confirmed by arithmetic. #29 gains bootstrap CIs. #34 closed. #41 inverted. #50 closed. #45–#49 added. Audit closed; no blocking unknowns remain. |
```

## Not detectable by grep

- `REPRODUCING.md` Table 9 entry marks the whole table **V** while verifying only the two means from the long CSV. It never checked the SD column or the three multi-run rows. State what it covers.
- `scripts/verify_branch.sh` `IDENTITY` implements identity tokens plus the machine name, but its header claims to cover "local path". A path with no name in it passes. Consider adding a path shape and an archive-identifier pattern (`10\.5281/zenodo`, `arxiv\.org`).
- `PROVENANCE.md` needs sections A, B and F of `docs/PROVENANCE_ADDENDUM_v7_7.md` merged in.
