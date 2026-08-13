# Power analysis: minimum detectable effect by topic

## Part 1. Canonical single-run design

Per-fold expert-vs-auto WSS@95 differences from `bow_stats_results.json`, the canonical single-run 5-fold analysis. MDE computed at alpha=0.05 (two-sided), power=0.80, from the exact noncentral-t distribution, on n=5 fold values per topic.

At n=5 the normal approximation understates the MDE by about 26% of the exact value (the exact value is 1.342 times the approximation). The approximation is shown for comparison only.

| Topic | n_total | n_folds | Observed mean | 95% CI | SD | SE | MDE exact (80% power) | MDE normal approx |
|---|---|---|---|---|---|---|---|---|
| Statins | 2744 | 5 | +0.1248 | [+0.0720, +0.1776] | 0.0675 | 0.0302 | 0.1135 | 0.0846 |
| Opiods | 1772 | 5 | -0.0100 | [-0.1414, +0.0860] | 0.1511 | 0.0676 | 0.2542 | 0.1894 |
| ADHD | 803 | 5 | -0.0298 | [-0.2178, +0.1364] | 0.2282 | 0.1020 | 0.3838 | 0.2859 |

### What this answers

The observed Statins effect at this design is **+0.1248**, against an exact MDE of 0.1135. Compare that effect size against the MDE at each smaller topic:

- **Opiods** (exact MDE = 0.2542): a Statins-sized effect (+0.1248) would NOT have been detected at this topic's variance and fold count.
- **ADHD** (exact MDE = 0.3838): a Statins-sized effect (+0.1248) would NOT have been detected at this topic's variance and fold count.

If a Statins-sized effect would have been detectable at Opioids/ADHD given their variance, the absence of a gap at those topics is informative about the effect. If the MDE is larger than the Statins effect, this design alone does not distinguish an absent gap from an undetected one.

## Part 2. Pooled multi-run fold count

The same quantity at the fold count used in the main results tables. This treats the pooled per-fold differences as independent observations, which they are not: folds within a rerun share four fifths of their training data and all reruns share the corpus. The effective sample size lies between Part 1 and Part 2 and is not determined by this design. The mean and SD columns are the source of the design-sensitivity table's multi-run rows.

| Topic | n_runs | n_folds | Mean | SD | MDE exact (80% power) |
|---|---|---|---|---|---|
| Statins | 7 | 35 | +0.0957 | 0.0641 | 0.0313 |
| Opiods | 7 | 35 | +0.0066 | 0.1713 | 0.0835 |
| ADHD | 7 | 35 | +0.0059 | 0.1620 | 0.0790 |

Statins effect at this fold count: **+0.0957**. Report both parts together; neither on its own bounds the cross-topic null.

