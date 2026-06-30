#!/usr/bin/env bash
# ============================================================================
# run_statins_subsampling.sh
# ----------------------------------------------------------------------------
# Experiment A: BoW Statins subsampled to ADHD-matched n=803, 7 subsample
# seeds, 4 text modes, 5-fold stratified CV per run.
#
# Answers Christer Q3/Q6: is the Statins gap a function of corpus size, or
# of lexical-conceptual structure? At matched n, if the gap persists, the
# size explanation is ruled out. If it collapses, §5.2 needs reframing.
#
# Outputs:
#   paper_experiments/outputs/bow_statins_subN803_subseed{1..7}_modes.txt
#
# Idempotent: skips runs whose output file already exists. Re-run after a
# Ctrl-C to resume from where you stopped.
#
# Usage (from repo root, with .venv312 active):
#   bash paper_experiments/run_statins_subsampling.sh you@example.com
#
# Or set EMAIL in the environment:
#   EMAIL=you@example.com bash paper_experiments/run_statins_subsampling.sh
# ============================================================================

set -u

EMAIL="${EMAIL:-${1:-}}"
if [ -z "$EMAIL" ]; then
  echo "Usage: bash $0 your.email@example.com"
  echo "   or: EMAIL=your.email@example.com bash $0"
  exit 1
fi

# ADHD has 803 abstracts after PubMed filter (per Consolidation v2 §2).
# Match Statins to that count to isolate corpus-size from topic-effect.
SUBSAMPLE_N=803
SEEDS=(1 2 3 4 5 6 7)
TOPIC="Statins"
KFOLD=5
WORKFLOW=8

OUTPUT_DIR="paper_experiments/outputs"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/run_statins_subsampling_$(date +%Y%m%d_%H%M%S).log"
echo "Starting BoW Statins subsampling experiment at $(date)"  | tee "$LOG_FILE"
echo "Topic: $TOPIC, subsample_n=$SUBSAMPLE_N, kfold=$KFOLD, workflow=$WORKFLOW" | tee -a "$LOG_FILE"
echo "Seeds: ${SEEDS[*]}"                                       | tee -a "$LOG_FILE"
echo "Email: $EMAIL"                                            | tee -a "$LOG_FILE"
echo ""                                                         | tee -a "$LOG_FILE"

session_start=$(date +%s)
done_count=0
total=${#SEEDS[@]}

for seed in "${SEEDS[@]}"; do
  out_file="$OUTPUT_DIR/bow_statins_subN${SUBSAMPLE_N}_subseed${seed}_modes.txt"

  if [ -f "$out_file" ]; then
    echo "[skip] $out_file already exists" | tee -a "$LOG_FILE"
    done_count=$((done_count + 1))
    continue
  fi

  t0=$(date +%s)
  echo "[$((done_count + 1))/$total] subseed=$seed -> $out_file" | tee -a "$LOG_FILE"

  python -m src.cohen_pipeline \
    --topic "$TOPIC" \
    --email "$EMAIL" \
    --compare-text-modes \
    --workflow "$WORKFLOW" \
    --kfold "$KFOLD" \
    --subsample-n "$SUBSAMPLE_N" \
    --subsample-seed "$seed" \
    --output-file "$out_file" 2>&1 | tail -5 | tee -a "$LOG_FILE"

  t1=$(date +%s)
  echo "       complete in $((t1 - t0))s" | tee -a "$LOG_FILE"
  done_count=$((done_count + 1))
done

elapsed=$(( $(date +%s) - session_start ))
echo ""                                                                      | tee -a "$LOG_FILE"
echo "Done. $done_count/$total runs complete. Total wall time: ${elapsed}s." | tee -a "$LOG_FILE"
echo "Outputs in: $OUTPUT_DIR"                                               | tee -a "$LOG_FILE"
echo ""                                                                      | tee -a "$LOG_FILE"
echo "Next step:"                                                            | tee -a "$LOG_FILE"
echo "  python paper_experiments/parse_bow_experiments.py"                   | tee -a "$LOG_FILE"
