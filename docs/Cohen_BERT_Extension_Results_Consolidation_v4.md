# Cohen BERT Extension — Results Consolidation v4

**Supersedes:** v3 (June 26, 2026)
**Date:** June 30, 2026
**Status:** Closed unless new experiments performed.

This document consolidates the BoW multi-run, BiomedBERT multi-seed, and design-sensitivity experiments for the Cohen et al. (2006) drug-class benchmark extension. It is the single empirical source of truth for paper_draft_v5.tex and supersedes Consolidation v1, v2, and v3.

---

## §1. What changed from v3

Three additions in v4:

1. **Design-sensitivity analyses on Statins (§6, new).** Two robustness experiments completed June 30, 2026:
   - Stratified subsampling of Statins to ADHD-matched n=803, seven subsample seeds × five folds × four text modes
   - Switch from 5-fold to 10-fold cross-validation at full Statins n=2,744, seven reruns
2. **Empirical power analysis (§6.3, new).** Minimum detectable effect at each topic's per-fold variance, given the canonical 5-fold design.
3. **Revised conclusions (§7).** The "topic-stratified absorption" reading of v3 is restated as design-conditional. The cross-topic null at Opioids/ADHD is reclassified as design-limited rather than informative.

The doc-drift hypothesis that v1-v3 carried about a CountVectorizer-patched features.py was investigated and **falsified**. The BoW pipeline runs on the Keras Tokenizer version of `src/features.py` as committed on GitHub main. N-grams enter through `preprocessing.py` as underscore-joined tokens. The "patched features.py" language in v1-v3 §9.5 should be read as documentation drift and ignored.

---

## §2. Empirical headline numbers

**Canonical evaluation (5-fold, full corpus size):**

| Classifier | Topic | Mean diff | 95% bootstrap CI | n folds |
|---|---|---|---|---|
| BoW (7-rerun multi-run) | Statins | **+0.096** | [+0.075, +0.116] | 35 |
| BoW (7-rerun multi-run) | Opioids | +0.007 | [−0.050, +0.061] | 35 |
| BoW (7-rerun multi-run) | ADHD | +0.006 | [−0.046, +0.060] | 35 |
| BoW pooled across topics | — | +0.036 | [+0.009, +0.063] | 105 |
| BERT (5-seed multi-seed) | Statins | +0.020 | [−0.011, +0.052] | 25 |
| BERT (5-seed multi-seed) | Opioids | −0.048 | [−0.116, +0.020] | 25 |
| BERT (5-seed multi-seed) | ADHD | +0.003 | [−0.040, +0.046] | 25 |
| BERT pooled across topics | — | −0.008 | [−0.039, +0.024] | 75 |

**Design-sensitivity on Statins (new in v4):**

| Design | Mean diff | 95% bootstrap CI | n folds |
|---|---|---|---|
| 5-fold, full n=2,744 (canonical) | **+0.096** | [+0.075, +0.116] | 35 |
| **5-fold, subsampled n=803 (matched-ADHD)** | **+0.033** | **[−0.022, +0.092]** | **35** |
| **10-fold, full n=2,744** | **+0.021** | **[+0.001, +0.041]** | **70** |

**Power analysis (new in v4):**

| Topic | Observed mean diff | Per-fold SD | MDE at 80% power |
|---|---|---|---|
| Statins | +0.125 | 0.068 | 0.085 (detectable) |
| Opioids | −0.010 | 0.151 | 0.189 (1.5× Statins effect) |
| ADHD | −0.030 | 0.228 | 0.286 (2.3× Statins effect) |

---

## §3. The four claims the data support

In order of strength:

1. **The expert-vs-auto MeSH gap is robust on Statins under the canonical Cohen evaluation design.** Multi-run mean +0.096, all seven reruns positive, 32 of 35 per-fold gaps positive, bootstrap CI excludes zero, permutation p < 0.001. This is a real effect at a real evaluation design.

2. **The Statins gap is conditional on the canonical design.** At matched corpus size n=803, the gap drops to +0.033 (CI includes zero). At 10-fold instead of 5-fold (full n), it drops to +0.021 (CI just excludes zero). The canonical-design effect of +0.096 attenuates roughly 3-5× under both perturbations.

3. **BiomedBERT under canonical evaluation produces a Statins gap comparable to BoW under 10-fold evaluation.** BERT 5-fold: +0.020. BoW 10-fold: +0.021. Within sampling noise of each other. Both an order of magnitude below BoW 5-fold reference. This reframes the "absorption" claim of v3.

4. **The Opioids/ADHD nulls are design-limited rather than informative.** Empirical MDE at those topics' variances exceeds the Statins effect size. A Statins-sized effect at those topics would not have been detected with the present sample sizes. We cannot conclude "no effect there" — only "no effect detectable under the present design."

---

## §4. The two readings the discussion offers

The data are consistent with two not-mutually-exclusive interpretations. Both belong in §5.1 of the paper.

**Reading A — Real effect, modulated by classifier effective training volume.**
The expert-vs-auto MeSH gap is real and structural. The classifier's response to it depends on how effectively it can extract signal from the abstract alone. Limited classifier or limited training data → MeSH augmentation matters → gap observable. More capable classifier or more training data → abstract supplies more signal → gap shrinks. Explains 10-fold BoW (+0.021) and 5-fold BERT (+0.020) by the same mechanism: both have more effective access to the abstract than 5-fold BoW.

**Reading B — Canonical estimate is high; true effect closer to 10-fold magnitude.**
The 5-fold full-n result is the point estimate most cleanly recovered from the Cohen benchmark but is not the typical operational value. The CIs of canonical and 10-fold do not overlap, so this is a real difference between designs rather than sampling noise. Under this reading, the literature's reported ~0.10 WSS@95% expert-MeSH advantage on Statins is an overestimate of the typical effect.

Paper does not adjudicate. Both readings produce a publishable methodological contribution: design-sensitivity of a benchmark result that has been treated as design-invariant.

---

## §5. What the paper claims now versus what v3 claimed

| Claim | v3 (June 26) | v4 (June 30) |
|---|---|---|
| Statins canonical-design effect | "robust, +0.096" | unchanged |
| Cross-topic pattern | "topic-stratified" | "topic-stratified under canonical design only" |
| BERT absorption | "BERT absorbs the Statins gap" | "BERT under canonical evaluation gives a result comparable to BoW under designs that increase per-fold training volume" |
| Opioids/ADHD null | "the gap does not exist at these topics" | "the gap is not detectable at these topics under the present design; whether it exists is open" |
| Lexical-semantic frame | "explains the Statins finding" | "remains relevant as input-difference description; observability is design-conditional" |
| Operational implication | "BERT with auto-MeSH ≈ BERT with expert-MeSH" | unchanged; magnitude bounded by 0.02 |

---

## §6. Design-sensitivity experimental detail

### §6.1 Subsampling protocol

Stratified subsampling of the Statins DataFrame after PubMed loading, preserving 5.5% inclusion rate. Uses `sklearn.model_selection.train_test_split` with `stratify=df["labels"]`. Seven subsample seeds (1 through 7) generate seven independent matched-size subsamples of n=803 each. Each subsample runs through the standard 4-mode × 5-fold pipeline, producing 35 per-fold expert-vs-auto paired differences total.

Per-fold counts per condition: ~9 included articles per test fold (compared to ~30 at 5-fold full-n).

Runtime: 7 × 4 modes × 5 folds × ~10 minutes per fold ≈ 50 minutes wall-clock total on local Windows CPU.

### §6.2 10-fold protocol

Statins at full n=2,744 with `--kfold 10` instead of `--kfold 5`. Seven reruns to match the existing multi-run protocol structure. Same pipeline, same data, same `SPLIT_SEED`, only the fold count differs.

Per-fold counts: ~15 included articles per test fold (compared to ~30 at 5-fold full-n).

Runtime: 7 × 4 modes × 10 folds ≈ 33 minutes per rerun on local Windows CPU, ~3.8 hours total.

### §6.3 Power analysis

Computed from the canonical 5-fold per-fold variance distributions, available as `bow_stats_results.json["{topic}"]["diffs"]`. MDE formula: `(z_{1-α/2} + z_{power}) × SD / sqrt(n)`. With α=0.05 two-sided, power=0.80, n=5 folds, normal approximation. T-correction shifts MDE by < 4% at n=5.

### §6.5 Token-length audit (added July 1, 2026)

BiomedBERT tokenizer applied to the three topics' input texts under each of the three non-auto-mesh text modes. Truncation at 512 tokens disclosed in paper §3.3, §5.1, and §6.

| Topic | Mode | n | Median tokens | Max | Truncated@512 |
|---|---|---|---|---|---|
| Statins | abstract | 2744 | 312 | 1929 | 167 (6.1%) |
| Statins | title_abstract | 2744 | 331 | 1947 | 218 (7.9%) |
| **Statins** | **title_abstract_mesh** | **2744** | **381** | **1982** | **415 (15.1%)** |
| Opioids | abstract | 1772 | 291 | 1075 | 83 (4.7%) |
| Opioids | title_abstract | 1772 | 308 | 1095 | 103 (5.8%) |
| **Opioids** | **title_abstract_mesh** | **1772** | **357** | **1163** | **184 (10.4%)** |
| ADHD | abstract | 803 | 294 | 986 | 37 (4.6%) |
| ADHD | title_abstract | 803 | 313 | 1002 | 44 (5.5%) |
| **ADHD** | **title_abstract_mesh** | **803** | **363** | **1046** | **95 (11.8%)** |

Reading: BiomedBERT sees less of the MeSH augmentation precisely at the topic-mode combination where MeSH augmentation matters most for the bag-of-words classifier (Statins title_abstract_mesh). This adds a third candidate explanation alongside the training-volume and canonical-overestimate readings in §4. Disclosed in paper §3.3 + §5.1 + §6 rather than treated as a confound.

Output files (under paper_experiments/outputs/):
- `audit_token_lengths.json` — raw per-topic-mode statistics
- `audit_token_lengths.md` — paper-quality table

### §6.6 What we did NOT run

- 10-fold at matched n=803 (joint design). Per-fold included-article count would drop to ~4, too low for WSS@95% stability.
- BERT under 10-fold or matched-n. Decision made June 30 after seeing BoW result: the BoW subsampling shows the gap collapses without needing BERT confirmation, and the BoW 10-fold result is already within sampling noise of canonical BERT.
- Variable training-set-fraction learning curve. Noted as future work in §5.5.

---

## §7. Production-facing artifacts

All files committed to `SamInMotion/Medical-intervention-text-classification` main branch as of June 30, 2026 commit (see `commit_and_store.sh` log for SHA):

**Scripts (paper_experiments/):**
- `patch_cohen_pipeline.py` — adds `--subsample-n` / `--subsample-seed` flags to BoW pipeline
- `run_statins_subsampling.sh` — Experiment A driver
- `run_statins_10fold.sh` — Experiment B driver
- `parse_bow_experiments.py` — bootstrap CI and decision verdict
- `power_analysis.py` — MDE table
- `audit_token_lengths.py` — BERT token-truncation analysis (pending, low priority)
- `audit_bow_bert_data_parity.md` — narrative audit

**Outputs (paper_experiments/outputs/):**
- `bow_statins_subN803_subseed{1..7}_modes.txt` — Experiment A raw output
- `bow_statins_kfold10_run{1..7}_modes.txt` — Experiment B raw output
- `bow_experiments_summary.csv` — long-format per-fold values
- `bow_experiments_summary.md` — bootstrap CIs table
- `bow_experiments_decision.txt` — verdict text
- `power_analysis.md` — MDE table

**Existing outputs from v3 (unchanged):**
- `outputs/bow_{topic}_multirun_summary.json` — canonical 5-fold BoW data
- `outputs/bert_{topic}_multiseed_summary.json` — BERT data
- `outputs/bert_three_topic_multiseed_summary.json` — pooled BERT
- `fig1_gap_forest.pdf` and `.png` — figure 1 (unchanged in v5)

**Paper:**
- `paper/paper_draft_v5.tex` — current manuscript (supersedes v4)

---

## §8. Open items for future sessions

| Item | Trigger | Owner |
|---|---|---|
| Venue selection | After §5.2 revision lands | Sam, 30-min publikasjonskanaler scan |
| Christer reply | After July (his vacation ends) | Sam, Mode A acknowledgement |
| audit_token_lengths.py run | Optional, only if §3.5 BERT truncation note needs numbers | Sam, ~5 min in venv |
| 10-fold matched-n joint analysis | If reviewer requests | Sam, ~2 hours compute, may not produce stable WSS@95 |
| BERT 10-fold confirmation | If reviewer requests | Sam, ~3 hours Colab T4 |
| BoW learning-curve at varying training-fraction | If review wants design-isolation cleaner | Sam, ~6 hours compute |
| Cohen topics 4-15 extension | If review wants cross-topic generality | Sam, days of compute |

None of these are blockers for submission.

---

## §9. Provenance

| Decision | Date | Reference |
|---|---|---|
| H-Pub2 registered | June 18 | Phase_0_Positioning_Note_v1 §3 |
| BERT multi-seed Statins completed | June 18 | Consolidation v1 §4 |
| BERT multi-seed Opioids/ADHD completed | June 26 | Consolidation v3 §4.3 |
| Paper draft v4 sent to Christer Johansson | June 29 | Sam's email 01:39 WEST |
| Christer reply received | June 29 | 13:54 WEST, six methodology questions |
| Design-sensitivity experiments completed | June 30 | This document §6 |
| Decision: BERT subsampling NOT needed | June 30 | bow_experiments_decision.txt |
| Paper draft v5 produced | June 30 | This document |

End of consolidation v4.
