# Context Update 188

**Date:** June 30, 2026
**Session:** Paper revision following Christer Johansson reply
**Supersedes:** CU 187 §6 (paper-state), CU 187 §8.5 partial (item 17 closed)
**Status:** Open items reduced to TU Wien (July 9) and venue scout

## §1. What this session did

Resolved Christer Johansson's six methodology questions on paper_draft_v4 through three new analyses, then revised the paper to incorporate the findings.

Computations completed on local Windows in `.venv312`:

1. **BoW Statins subsampling experiment** (Experiment A) — Statins stratified-subsampled to ADHD-matched n=803, 7 subsample seeds × 5 folds × 4 text modes. Result: gap +0.033, 95% bootstrap CI [−0.022, +0.092], includes zero.

2. **BoW Statins 10-fold sensitivity** (Experiment B) — Statins at full n=2,744 with 10-fold instead of 5-fold, 7 reruns. Result: gap +0.021, 95% bootstrap CI [+0.001, +0.041], just excludes zero.

3. **Power analysis** — Empirical MDE per topic. Statins MDE=0.085 (detectable), Opioids MDE=0.189, ADHD MDE=0.286. Cross-topic null at Opioids/ADHD is design-limited.

Decision: **BERT subsampling confirmation NOT run.** The BoW result is sufficient. Colab T4 freed for other use.

## §2. What was decided

The §5.2 semasiological/onomasiological framing of v3/v4 cannot stand as the primary explanation. The revised framing in paper_draft_v5:

- Statins effect at canonical 5-fold full-n is real (+0.096, robust across seven reruns)
- This effect is **conditional on the evaluation design**: it shrinks 3× under matched-n subsampling and 5× under 10-fold cross-validation
- BERT under canonical evaluation gives a Statins gap (+0.020) **within sampling noise** of BoW under 10-fold (+0.021)
- The cross-topic null at Opioids/ADHD is design-limited — empirical MDE exceeds the Statins effect, so we cannot distinguish "no effect there" from "underpowered to detect"
- The lexical-semantic distinction remains relevant as an input-difference description; its **observability in classifier outputs is design-conditional**
- Operational implication for screening pipelines: BERT with auto-MeSH performs within ~0.02 WSS@95% of BERT with expert-MeSH — bounded, viable substitute

The data support a publishable contribution. The reframing makes the paper stronger by anticipating the strongest reviewer attack (design-sensitivity) with empirical evidence rather than prose hedging.

## §3. Files produced this session

All under `paper_experiments/` in the local repo (also pushed to GitHub main and copied to Drive):

- `verify_setup.sh`, `state_diagnostic.sh`, `local_inspect.sh` — diagnostics (one-shot, retained for reproducibility)
- `patch_cohen_pipeline.py` — applies subsampling flags to BoW pipeline; idempotent; produces `.bak`
- `run_statins_subsampling.sh` — Experiment A driver (v2 with proper error handling)
- `run_statins_10fold.sh` — Experiment B driver (v2)
- `parse_bow_experiments.py` — bootstrap CI parser and verdict generator
- `power_analysis.py` — MDE table generator
- `audit_token_lengths.py` — BERT token-truncation analysis (not yet run, low priority)
- `audit_bow_bert_data_parity.md` — narrative audit covering Christer Q1

Outputs under `paper_experiments/outputs/`:

- 14 raw `.txt` files from the two experiments
- `bow_experiments_summary.csv`, `bow_experiments_summary.md`, `bow_experiments_decision.txt`
- `power_analysis.md`
- Run logs

Paper:

- `paper/paper_draft_v5.tex` — replaces v4. Adds §3.6 (Robustness analyses), §4.4 (Evaluation design sensitivity), reframes §5.1 and §5.2.

Documentation:

- `Cohen_BERT_Extension_Results_Consolidation_v4.md` — single source of truth for paper findings, supersedes v3

## §4. Pipeline state, unchanged from CU 187 except where noted

**PhD pipeline:**
- TU Wien Hoyle Prae-Doc, deadline **July 9, 9 days** — **REMAINS P0**, this session did not displace it
- TUM Ziegelmayer #164 — submitted June 29
- Touileb #163 — under evaluation
- Leiden LIACS #16571 — submitted June 25, cold
- Grieg #120 — Rank 2, conditional
- Rejected: SLATE #117, HVL #130, LMLM #122, ProtAIomics #91

**Industry pipeline:** Karnov #158, BASE #178, Luxembourg #165 submitted. Sensio #184 awaiting posting. Rejected: Aker BP June 26, Avelios June 28, Volve.

**Combined "at least one offer" probability:** ~56% per CU 187. Unchanged by this session.

## §5. Open items for next session

Sorted by priority:

1. **TU Wien Hoyle motivation letter** — Sam writing from scratch per no-AI constraint. Claude role is structural scaffolding + post-draft check. Deadline July 9. **P0.**
2. **Venue scout** — 30 min publikasjonskanaler scan. Candidate MDPI venues (Information, Applied Sciences) plus BMC Medical Informatics. After TU Wien.
3. **Christer reply** — Mode A, after his vacation (after July). Draft text in CU 186 §"Christer reply".
4. **paper_draft_v5 PDF compile + final read-through** — pdflatex two-pass. After TU Wien.
5. **Phase 0 Positioning Note v5** — if v4 framing affects PhD positioning (Leiden uses paper-in-prep as a credential signal). Read v4 first, decide if v5 needed. Low urgency.
6. **Audit token-lengths run** — 5 minutes when venv is active. Optional, only if reviewer asks about truncation.

## §6. What is closed by this session

- ✅ Christer's six methodology questions — answered by Experiments A and B + power analysis
- ✅ Paper draft v5 — incorporates findings
- ✅ "Patched features.py" doc-drift hypothesis from Consolidation v1/v2/v3 — investigated and falsified, removed from v4
- ✅ Subscription continuity plan §17 in CU 187 — partially addressed by Consolidation v4 acting as the canonical paper-state document and this CU 188 closing the session record. Three-document handoff package collapses to two: Consolidation v4 + this CU 188. The decision rules cheatsheet and outcome decision tree remain queued.

## §7. What the next Claude session should know on entry

If Claude starts a new conversation and needs to catch up on paper state:

1. Read **Cohen_BERT_Extension_Results_Consolidation_v4.md** first. That is the source of truth for paper findings.
2. Read **this CU 188** for what was decided and what is queued.
3. Then read **paper_draft_v5.tex** if revising the manuscript.
4. The userMemories will be stale on these items — trust the consolidation and CU over userMemories per Rule Zero.
5. `paper_experiments/RESULTS_SUMMARY.md` is a lightweight pointer if the consolidation feels long.

If Claude is asked about the paper "what does it say":
- Canonical Statins +0.096 holds, with design-sensitivity caveat
- BERT result +0.020 reframed as "comparable to BoW under 10-fold"
- Cross-topic null is design-limited, not informative
- §5.2 lexical-semantic remains relevant but observability is conditional

If Claude is asked to run more experiments:
- Subsampling and 10-fold already done. Don't re-run.
- BERT subsampling NOT needed — decision made June 30.
- Optional: audit_token_lengths.py for §3.5 truncation numbers.

End of CU 188.
