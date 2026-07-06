# scripts/ — Curated Script Archive

This directory contains **historical, superseded, and figure-generation scripts**
that are preserved for provenance but are **not part of the canonical
reproducibility pipeline**. The canonical scripts live at the repository root.

## Files and their status

| File | Status | Reason |
|------|--------|--------|
| `make_fig1_gap_forest_v3.py` | **Historical** | Superseded by `paper_experiments/make_fig1_gap_forest_v3.py` or root-level figure generation. Retained because it generated the forest plot in earlier paper versions. |
| `make_fig2_design_sensitivity.py` | **Historical** | Generated Figure 2 (design sensitivity) in earlier versions. Retained for provenance. |
| `make_fig1_v2.py` | **Superseded** | Earlier forest plot generator that used hardcoded BERT values. Superseded by v3. |
| `make_paper_artifacts.py` | **Superseded** | Pre-audit baseline that assumed filenames without `_seed42` suffix. The audit-aware patch is documented but not executed; canonical paths run through `bootstrap_paired_permutation.py`. |
| `run_bow_multirun.sh` | **Orchestrator** | Shell wrapper for running the BoW multi-run protocol. Retained for batch-execution provenance. |

## Canonical paths (active)

- Root: `bootstrap_paired_permutation.py`, `parse_bow_multirun.py`, `demo_statistical_analysis.py`
- `paper_experiments/`: `power_analysis.py`, `patch_cohen_pipeline.py`, `run_statins_subsampling.sh`, `run_statins_10fold.sh`, `parse_bow_experiments.py`

## When to use this directory

- **Never** for reproducing the paper's current claims.
- **Only** if you are tracing the evolution of a figure or investigating a historical analysis path.
