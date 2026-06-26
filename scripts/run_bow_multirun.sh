#!/usr/bin/env bash
# run_bow_multirun.sh
#
# Multi-run characterisation of BoW pipeline on Opioids and ADHD.
# Mirrors the Statins multi-run protocol (CU 177, paper draft v2/v3 §4.1):
# seven reruns per topic with identical command-line and identical seed
# (set_seeds(42) inside the pipeline). Variance comes from Keras Dense layer
# init non-determinism on the current Windows TensorFlow build (H-Repro1
# falsified the oneDNN hypothesis; H-Repro2 is the remaining candidate).
#
# Usage from repo root (in MinGW64 / Git Bash on Windows):
#     source .venv312/Scripts/activate
#     export TF_ENABLE_ONEDNN_OPTS=0
#     bash scripts/run_bow_multirun.sh
#
# Override the email or topic list via environment variables if needed:
#     EMAIL=you@example.com bash scripts/run_bow_multirun.sh
#
# Outputs (in outputs/, paths relative to repo root):
#     bow_opiods_run{1..7}.txt
#     bow_adhd_run{1..7}.txt
#
# Expected runtime on a typical Windows laptop:
#     Opioids (1,772 abstracts): ~3-5 min per run × 7 ≈ 25-35 min
#     ADHD    (803 abstracts):   ~2-3 min per run × 7 ≈ 15-20 min
#     Total: ~45-60 min

set -euo pipefail

EMAIL="${EMAIL:-sammy.okmens@gmail.com}"
TOPICS=("Opiods" "ADHD")   # cache spelling for Opioids preserved by Cohen TSV
RUNS=7

# Sanity check: cwd should be the repo root (where src/ and outputs/ live).
if [[ ! -d "src" ]] || [[ ! -d "outputs" ]]; then
    echo "ERROR: Run this script from the repo root, where src/ and outputs/ exist."
    echo "Current directory: $(pwd)"
    exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "WARNING: no virtual environment active."
    echo "Recommended: source .venv312/Scripts/activate before running this script."
    echo
fi

if [[ "${TF_ENABLE_ONEDNN_OPTS:-}" != "0" ]]; then
    echo "NOTE: TF_ENABLE_ONEDNN_OPTS not set to '0'. Setting for this script."
    export TF_ENABLE_ONEDNN_OPTS=0
fi

session_start=$(date +%s)

for topic in "${TOPICS[@]}"; do
    topic_lower=$(echo "$topic" | tr '[:upper:]' '[:lower:]')
    echo "================================================================"
    echo "TOPIC: $topic ($RUNS runs)"
    echo "================================================================"
    topic_start=$(date +%s)

    for i in $(seq 1 $RUNS); do
        output_file="outputs/bow_${topic_lower}_run${i}.txt"

        if [[ -f "$output_file" ]]; then
            echo "[skip] $output_file already exists"
            continue
        fi

        echo "  Run $i/$RUNS for $topic ..."
        run_start=$(date +%s)

        # Pipeline command: identical to the Statins multi-run command,
        # which produced bow_statins_run1.txt etc. The --compare-text-modes
        # flag runs all four text modes (abstract, title_abstract,
        # title_abstract_mesh, auto_mesh) in one invocation and writes the
        # combined output to --output-file. The parser locates each mode
        # section by the "Cohen topic: <Topic> (<mode> mode)" header.
        python -m src.cohen_pipeline \
            --topic "$topic" \
            --email "$EMAIL" \
            --compare-text-modes \
            --output-file "$output_file"

        run_elapsed=$(($(date +%s) - run_start))
        echo "  Run $i/$RUNS for $topic complete in ${run_elapsed}s"
        echo
    done

    topic_elapsed=$(($(date +%s) - topic_start))
    echo "Topic $topic complete in ${topic_elapsed}s"
    echo
done

session_elapsed=$(($(date +%s) - session_start))
echo "================================================================"
echo "ALL DONE. Total runtime: ${session_elapsed}s"
echo "================================================================"
echo
echo "Outputs:"
ls -lh outputs/bow_opiods_run*.txt 2>/dev/null || echo "  (no opiods outputs)"
ls -lh outputs/bow_adhd_run*.txt 2>/dev/null || echo "  (no adhd outputs)"
echo
echo "Next: parse the runs into summary JSONs"
echo "  python scripts/parse_bow_multirun.py --topic Opiods"
echo "  python scripts/parse_bow_multirun.py --topic ADHD"
echo "  # or in one go for all three topics:"
echo "  python scripts/parse_bow_multirun.py --all"
echo
echo "Then sync to Drive:"
echo "  cp outputs/bow_{opiods,adhd}_run*.txt /g/My\\ Drive/cohen_bert_run/outputs/"
echo "  cp outputs/bow_{opiods,adhd}_multirun_summary.json /g/My\\ Drive/cohen_bert_run/outputs/"
