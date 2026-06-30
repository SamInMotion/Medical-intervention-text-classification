# BoW Experiments Summary

Pooled per-fold expert-vs-auto WSS@95 gap, 95% percentile bootstrap CIs (10,000 resamples).

| Condition | n folds | n runs | Mean gap | 95% CI | SD |
|---|---|---|---|---|---|
| Statins full (n≈2,744), 5-fold (reference) | 5 | — | +0.1248 | [+0.0720, +0.1776] | 0.0675 |
| Statins subsampled (n=803), 5-fold (NEW) | 35 | 7 | +0.0332 | [-0.0215, +0.0922] | 0.1774 |
| Statins full (n≈2,744), 10-fold (NEW) | 70 | 7 | +0.0207 | [+0.0007, +0.0405] | 0.0859 |
| ADHD full (n≈803), 5-fold (reference) | 5 | — | -0.0298 | [-0.2178, +0.1364] | 0.2282 |
| Opioids full (n≈1,772), 5-fold (reference) | 5 | — | -0.0100 | [-0.1414, +0.0860] | 0.1511 |

## Files parsed

Subsampling experiment:
  - bow_statins_subN803_subseed1_modes.txt
  - bow_statins_subN803_subseed2_modes.txt
  - bow_statins_subN803_subseed3_modes.txt
  - bow_statins_subN803_subseed4_modes.txt
  - bow_statins_subN803_subseed5_modes.txt
  - bow_statins_subN803_subseed6_modes.txt
  - bow_statins_subN803_subseed7_modes.txt

10-fold experiment:
  - bow_statins_kfold10_run1_modes.txt
  - bow_statins_kfold10_run2_modes.txt
  - bow_statins_kfold10_run3_modes.txt
  - bow_statins_kfold10_run4_modes.txt
  - bow_statins_kfold10_run5_modes.txt
  - bow_statins_kfold10_run6_modes.txt
  - bow_statins_kfold10_run7_modes.txt
