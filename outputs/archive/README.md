# outputs/archive/

Superseded or exploratory output files preserved as experimental record.
Not cited by the paper. Useful for understanding how the analysis evolved.

## Files

### Pre-audit BERT analysis

- `analysis_results_full.json` (June 12, 2026) — Pre-audit BERT three-topic
  paired statistical analysis. Computed from implicit-RNG runs of
  `cohen_bert_pipeline.py` before `--seed 42` was passed explicitly.
  Superseded by `../analysis_results_full_v2.json`. Per-fold drift between
  the two runs is documented in `../audit_comparison.json`.

### BoW Statins reproducibility smoke tests (June 18-20)

These three files track the falsified oneDNN hypothesis that motivated
the seven-run multi-run characterisation.

- `bow_statins_smoke.txt` (June 18) — First BoW Statins run in the new
  Python 3.12 venv. Values drifted 0.004-0.012 WSS from the originally
  published thesis numbers. Triggered the oneDNN hypothesis.

- `bow_statins_smoke_onednn_off.txt` (June 20) — Second smoke test with
  `TF_ENABLE_ONEDNN_OPTS=0`. Was expected to reproduce the published
  numbers exactly if the oneDNN hypothesis was correct.

- `bow_statins_smoke_rerun2.txt` (June 20) — Third smoke test, identical
  command to the second, also with oneDNN off. Per-fold WSS values
  differed from the previous run by up to 0.033. Falsified the oneDNN
  hypothesis. The actual source of non-determinism was Keras layer
  initialisation that the pipeline's set_seeds function does not fully
  control. This file is the empirical pivot that motivated
  bow_statins_run{1..7}.txt in the parent directory.

### April thesis exploration

- `kfold_results.txt` (April 12) — Original thesis k-fold validation
  results on NEO classification (1,611 neurological concepts). The
  80-86% k-fold accuracy baseline the thesis rests on. Pre-Cohen
  benchmark.

- `all_workflows_statins.txt` (April 14) — April Cohen Statins workflow
  exploration. Multiple variant configurations tested before the BoW
  pipeline settled into the form documented in
  `paper/archive/cohen_benchmark_analysis.md`.

- `text_mode_comparison.txt` (April 14) — April text mode exploration.
  Side-by-side comparison of the four text modes prior to the
  structured benchmark pipeline.

### Early BERT validation

- `bert_val_tuned.txt` (June 11) — Early BiomedBERT validation tuning
  exploration during initial Colab setup. Orphan. Not cited by paper.
