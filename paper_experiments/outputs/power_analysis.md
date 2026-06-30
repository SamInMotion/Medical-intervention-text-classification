# Power analysis: minimum detectable effect by topic

Per-fold expert-vs-auto WSS@95 distributions from the BoW multi-run data give us topic-specific variance estimates. MDE computed at α=0.05 (two-sided), power=0.80, normal approximation. With n=35 fold values per topic the t-correction shifts MDE by under 4%.

| Topic | n_total | n_folds | Observed mean | 95% CI | SD | SE | MDE (80% power) |
|---|---|---|---|---|---|---|---|
| Statins | 2744 | 5 | +0.1248 | [+0.0720, +0.1776] | 0.0675 | 0.0302 | 0.0846 |
| Opiods | 1772 | 5 | -0.0100 | [-0.1414, +0.0860] | 0.1511 | 0.0676 | 0.1894 |
| ADHD | 803 | 5 | -0.0298 | [-0.2178, +0.1364] | 0.2282 | 0.1020 | 0.2859 |

## What this answers

The observed Statins effect is **+0.1248**. Compare this against the MDE at each smaller topic:

- **Opiods** (MDE = 0.1894): a Statins-sized effect (+0.1248) would NOT have been detected at this topic's n and variance.
- **ADHD** (MDE = 0.2859): a Statins-sized effect (+0.1248) would NOT have been detected at this topic's n and variance.

### Reading

If the Statins-sized effect *would* have been detectable at Opioids/ADHD given their variance, the absence of a gap at those topics is informative — it argues against the pure statistical-power explanation. If the MDE is *larger* than the Statins effect, we cannot tell from this design alone whether the gap is absent or merely undetectable.

This analysis is approximate. The matched-n subsampling experiment in `parse_bow_experiments.py` gives the direct answer for Statins specifically; the power analysis above is the complementary cross-topic check.
