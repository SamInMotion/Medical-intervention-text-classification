#!/usr/bin/env bash
# ============================================================================
# run_statins_10fold.sh
# ----------------------------------------------------------------------------
# Experiment B: BoW Statins at full n=2,744 with 10-fold instead of 5-fold,
# 7 reruns to match the existing 5-fold multi-run protocol structure.
#
# Answers Christer Q4: why 5-fold not 10? With this we say "we replicate
# at 10-fold" instead of "we follow Cohen convention."
#
# Per kfold_analysis.md, 10-fold at Statins is fine (~55 included articles
# per fold). Not run at Opioids/ADHD where per-fold instability would
# dominate. That asymmetry is documented in the limitations.
#
# Outputs:
#   paper_experiments/outputs/bow_statins_kfold10_run{1..7}_modes.txt
#
# Idempotent: skips runs whose output file already exists.
#
# Usage:
#   bash paper_experiments/run_statins_10fold.sh you@example.com
# ============================================================================

set -u

EMAIL="${EMAIL:-${1:-}}"
if [ -z "$EMAIL" ]; then
  echo "Usage: bash $0 your.email@example.com"
  exit 1
fi

KFOLD=10
RUNS=(1 2 3 4 5 6 7)
TOPIC="Statins"
WORKFLOW=8

OUTPUT_DIR="paper_experiments/outputs"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/run_statins_10fold_$(date +%Y%m%d_%H%M%S).log"
echo "Starting BoW Statins 10-fold sensitivity experiment at $(date)" | tee "$LOG_FILE"
echo "Topic: $TOPIC, kfold=$KFOLD, workflow=$WORKFLOW"                | tee -a "$LOG_FILE"
echo "Reruns: ${RUNS[*]}"                                              | tee -a "$LOG_FILE"
echo ""                                                                | tee -a "$LOG_FILE"

# The 7-rerun protocol relies on TF non-determinism. Subsampling is NOT
# applied here -- we keep full Statins at n=2,744 to mirror the existing
# 5-fold reference data exactly, varying only kfold.

session_start=$(date +%s)
done_count=0
total=${#RUNS[@]}

for run in "${RUNS[@]}"; do
  out_file="$OUTPUT_DIR/bow_statins_kfold${KFOLD}_run${run}_modes.txt"

  if [ -f "$out_file" ]; then
    echo "[skip] $out_file already exists" | tee -a "$LOG_FILE"
    done_count=$((done_count + 1))
    continue
  fi

  t0=$(date +%s)
  echo "[$((done_count + 1))/$total] run=$run -> $out_file" | tee -a "$LOG_FILE"

  python -m src.cohen_pipeline \
    --topic "$TOPIC" \
    --email "$EMAIL" \
    --compare-text-modes \
    --workflow "$WORKFLOW" \
    --kfold "$KFOLD" \
    --output-file "$out_file" 2>&1 | tail -5 | tee -a "$LOG_FILE"

  t1=$(date +%s)
  echo "       complete in $((t1 - t0))s" | tee -a "$LOG_FILE"
  done_count=$((done_count + 1))
done

elapsed=$(( $(date +%s) - session_start ))
echo ""                                                                       | tee -a "$LOG_FILE"
echo "Done. $done_count/$total runs complete. Total wall time: ${elapsed}s."  | tee -a "$LOG_FILE"
echo "Outputs in: $OUTPUT_DIR"                                                | tee -a "$LOG_FILE"
