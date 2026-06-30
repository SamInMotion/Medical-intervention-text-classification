# Paper experiments — workflow guide

Six computations addressing Christer Johansson's six questions on the paper draft. Designed to run primarily on local Windows (`.venv312`) with Colab T4 reserved for the conditional BERT confirmation in Phase 2.

## Files in this directory

| File | Purpose | Where it runs |
|---|---|---|
| `verify_setup.sh` | 30-second sanity check: local features.py matches GitHub, no CountVectorizer hiding on side branches | Git Bash, repo root |
| `patch_cohen_pipeline.py` | Adds `--subsample-n` / `--subsample-seed` flags to `src/cohen_pipeline.py` in place (creates `.bak` backup, idempotent) | Repo root, `.venv312` |
| `run_statins_subsampling.sh` | Experiment A: BoW Statins subsampled to ADHD-matched n=803, 7 subsample seeds × 4 modes × 5-fold | Local CPU, Git Bash |
| `run_statins_10fold.sh` | Experiment B: BoW Statins at full n with 10-fold instead of 5-fold, 7 reruns | Local CPU, Git Bash |
| `parse_bow_experiments.py` | Parses outputs from A and B, bootstrap CIs, writes verdict for §5.2 | `.venv312` |
| `power_analysis.py` | Analytical MDE from existing per-fold variance | `.venv312` |
| `audit_token_lengths.py` | Empirical BiomedBERT token-length distribution per topic | `.venv312` (needs `transformers`) |
| `audit_bow_bert_data_parity.md` | Code-reading audit, ready for §3.5/§6 once token-length numbers are in | Reference doc |

## Phase 0 — verify (5 minutes)

```bash
cd /c/Users/samue.KITAB/Medical-intervention-text-classification
bash paper_experiments/verify_setup.sh
```

Should print `[MATCH]` for both `features.py` and `preprocessing.py`, and either "no `features.py`" or "[OK] No CountVectorizer" for `v2.0-infastructure`. If any line says `[DIFFER]` or warns about CountVectorizer, stop and share the output before proceeding.

## Phase 1 — patch and smoke test (5 minutes)

```bash
# .venv312 activated
source .venv312/Scripts/activate   # or however you activate it

python paper_experiments/patch_cohen_pipeline.py
# Look for: [backup] src/cohen_pipeline.py.bak
#           [patched] src/cohen_pipeline.py

# Smoke test the patch (2 minutes)
python -m src.cohen_pipeline --topic Statins --email YOUR_EMAIL \
    --compare-text-modes --subsample-n 200 --subsample-seed 1 \
    --kfold 2 --epochs 5 \
    --output-file paper_experiments/outputs/smoke_test.txt

# Check the output
grep -E "abstracts|Fold" paper_experiments/outputs/smoke_test.txt | head -20
```

Expected: the output's "X-fold CV on N abstracts" line should show **200 abstracts** (the subsampled count), not 2,744. Two folds per text mode, four modes. If you see N=200 and four mode blocks each with two folds, the patch works.

## Phase 2 — main experiments in parallel (45-90 minutes)

Both scripts are idempotent. If you Ctrl-C and re-run, they skip completed runs.

```bash
# Terminal 1 — subsampling (the load-bearing one)
bash paper_experiments/run_statins_subsampling.sh YOUR_EMAIL

# Terminal 2 (optional, after Terminal 1 finishes, or different machine)
bash paper_experiments/run_statins_10fold.sh YOUR_EMAIL
```

Wall-clock estimates from the existing BoW multi-run timing on local Windows:
- Subsampling (smaller n=803): ~30-40 min total for all 7 subsample seeds
- 10-fold (full n=2,744 but twice the folds): ~50-70 min total for all 7 reruns

Running sequentially on one machine: ~80-110 minutes total. Running concurrently if you have the cores: ~50-70 minutes wall-clock.

## Phase 3 — parse and decide (5 minutes)

```bash
python paper_experiments/parse_bow_experiments.py
cat paper_experiments/outputs/bow_experiments_decision.txt
```

The decision file tells you which of three branches the experiment landed in. The threshold is +0.04 (roughly half the full-n Statins effect of +0.096):

1. **Gap persists, CI excludes zero, mean > 0.04** → §5.2 argument empirically defended. Proceed to BERT confirmation on Colab T4.
2. **Gap persists at attenuated magnitude (CI excludes zero, mean < 0.04)** → hedge §5.2. The effect survives but is weaker than at full n, consistent with mixed linguistic-structure + partial power.
3. **Gap collapses (CI includes zero)** → reframe §5.2. The original linguistic argument cannot be sustained; the paper is still publishable but with the power explanation as the leading reading.

## Phase 4 — supporting analyses (10 minutes)

Run alongside Phase 2 or after Phase 3.

```bash
# Power analysis (uses existing bow_stats_results.json)
python paper_experiments/power_analysis.py
cat paper_experiments/outputs/power_analysis.md

# Token-length audit (needs transformers and Cohen data)
python paper_experiments/audit_token_lengths.py --email YOUR_EMAIL
cat paper_experiments/outputs/audit_token_lengths.md
```

These two produce ready-to-paste markdown sections for §5.2 (power) and §3.5/§6 (truncation).

## Phase 5 — conditional BERT confirmation (Colab T4, ~2-3 hours)

Only run this if Phase 3 returned outcome (1) above. If (2) or (3), skip — the BoW result is sufficient and a BERT confirmation does not change the story.

The Colab notebook adaptation is small: clone the repo (which now contains the patched `cohen_pipeline.py`), use the same `--subsample-n 803 --subsample-seed S` flags on `cohen_bert_pipeline.py`. The BERT pipeline already accepts `--subsample N` as documented in its docstring, but the flag name differs from BoW — confirm by reading the BERT argparse block before running. I will produce the Colab cell when the decision is in.

## Mapping back to Christer's six questions

| Q | What addresses it |
|---|---|
| Q1 model/topic size mismatch | `audit_bow_bert_data_parity.md` + `audit_token_lengths.py` |
| Q2 data confound on the gap | `parse_bow_experiments.py` (subsampling result) + `power_analysis.py` |
| Q3 data confound at Statins | `parse_bow_experiments.py` (subsampling) — the direct answer |
| Q4 5-fold vs 10-fold | `run_statins_10fold.sh` + `parse_bow_experiments.py` |
| Q5 why not more data | Prose-only, addressed in §6 limitations |
| Q6 Statins improvement source | Same as Q3 — subsampling result |

Q5 needs no compute. Everything else lands in one of the scripts above.
