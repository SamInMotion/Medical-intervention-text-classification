#!/usr/bin/env bash
# ============================================================================
# commit_and_store.sh
# ----------------------------------------------------------------------------
# Single storage script for the June 30 paper revision session.
#
# Stores in three places:
#   1. Local repo (this directory) — paper, scripts, outputs, docs
#   2. GitHub remote (origin/main) — pushed via git
#   3. Google Drive /g/My Drive/cohen_bert_run/ — synced copies
#
# Pre-flight:
#   - Run from repo root
#   - Have the seven session artifacts in ~/Downloads or another known location
#     (the script asks if they are not at the default path)
#
# Usage:
#   bash paper_experiments/commit_and_store.sh
#
# Or with explicit source directory:
#   bash paper_experiments/commit_and_store.sh /c/Users/samue.KITAB/Downloads
# ============================================================================

set -u

SRC_DIR="${1:-$HOME/Downloads}"
DRIVE_DIR="/g/My Drive/cohen_bert_run"
REPO_ROOT=$(pwd)

# ---------- Pre-flight ----------
echo "============================================================"
echo "Pre-flight"
echo "============================================================"
echo "Repo root:       $REPO_ROOT"
echo "Session sources: $SRC_DIR"
echo "Drive target:    $DRIVE_DIR"
echo ""

if [ ! -d "$REPO_ROOT/.git" ]; then
  echo "[ABORT] Not in a git repo. cd to repo root first."
  exit 1
fi

if [ ! -d "$DRIVE_DIR" ]; then
  echo "[WARN] Drive directory not found at $DRIVE_DIR — Drive sync will be skipped."
  SKIP_DRIVE=1
else
  SKIP_DRIVE=0
fi

# The seven artifacts produced this session that must land in the repo
ARTIFACTS=(
  "paper_draft_v5.tex|paper/paper_draft_v5.tex"
  "Cohen_BERT_Extension_Results_Consolidation_v4.md|docs/Cohen_BERT_Extension_Results_Consolidation_v4.md"
  "Context_Update_188.md|docs/Context_Update_188.md"
)

# Check artifacts exist in source dir
echo "Checking session artifacts in $SRC_DIR ..."
missing=0
for spec in "${ARTIFACTS[@]}"; do
  src_name="${spec%%|*}"
  if [ ! -f "$SRC_DIR/$src_name" ]; then
    echo "  [MISSING] $SRC_DIR/$src_name"
    missing=$((missing + 1))
  else
    echo "  [found] $src_name"
  fi
done

if [ $missing -gt 0 ]; then
  echo ""
  echo "[ABORT] $missing artifact(s) not found at $SRC_DIR"
  echo "        Move the downloaded session files there, or pass the path as arg 1:"
  echo "        bash paper_experiments/commit_and_store.sh /path/to/files"
  exit 1
fi
echo ""

# ---------- Stage artifacts into repo ----------
echo "============================================================"
echo "Stage artifacts into repo"
echo "============================================================"

mkdir -p paper docs

for spec in "${ARTIFACTS[@]}"; do
  src_name="${spec%%|*}"
  dest_path="${spec##*|}"
  mkdir -p "$(dirname "$dest_path")"
  cp "$SRC_DIR/$src_name" "$dest_path"
  echo "  $src_name -> $dest_path"
done
echo ""

# ---------- Git status check ----------
echo "============================================================"
echo "Git status before commit"
echo "============================================================"
git status --short
echo ""

# ---------- Commit ----------
COMMIT_MSG="Paper revision v5 + design-sensitivity experiments

- Add paper_experiments/ with subsampling and 10-fold scripts and outputs
- Replace paper_draft_v4 with paper_draft_v5 (revised §5.1 and §5.2
  for design-conditional absorption; new §3.6 and §4.4 for
  evaluation design sensitivity)
- Add docs/Cohen_BERT_Extension_Results_Consolidation_v4.md
  (single source of truth for paper findings, supersedes v3)
- Add docs/Context_Update_188.md (session closeout)

Closes Christer Johansson methodology questions Q1-Q6 with
empirical evidence rather than prose hedging:
- Q3/Q6: subsampling shows Statins gap collapses at matched n
- Q4: 10-fold sensitivity shows gap shrinks ~5x vs 5-fold
- Q2: power analysis shows Opioids/ADHD nulls are design-limited
- Q1: data-parity audit in paper_experiments/

Decision: BERT subsampling confirmation NOT run; BoW result is
sufficient to reframe §5.2."

echo "============================================================"
echo "Commit"
echo "============================================================"
echo "Stage all changes:"
git add paper/ docs/ paper_experiments/
echo ""
echo "Files staged:"
git diff --cached --stat
echo ""
echo "Commit message preview:"
echo "------------------------------------------------------------"
echo "$COMMIT_MSG"
echo "------------------------------------------------------------"
echo ""

read -p "Proceed with commit? [y/N] " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
  echo "[abort] User cancelled. Files staged but not committed."
  exit 0
fi

git commit -m "$COMMIT_MSG"
commit_sha=$(git rev-parse HEAD)
echo ""
echo "Commit SHA: $commit_sha"
echo ""

# ---------- Push to GitHub ----------
echo "============================================================"
echo "Push to GitHub origin/main"
echo "============================================================"
git push origin main
push_status=$?
if [ $push_status -ne 0 ]; then
  echo "[WARN] Push failed (exit $push_status). The commit is local; run"
  echo "       'git push origin main' later when network is available."
else
  echo "Pushed to origin/main."
fi
echo ""

# ---------- Drive sync ----------
if [ $SKIP_DRIVE -eq 0 ]; then
  echo "============================================================"
  echo "Sync to Google Drive"
  echo "============================================================"

  # Copy the three docs and the paper to Drive root
  cp paper/paper_draft_v5.tex "$DRIVE_DIR/"
  cp docs/Cohen_BERT_Extension_Results_Consolidation_v4.md "$DRIVE_DIR/"
  cp docs/Context_Update_188.md "$DRIVE_DIR/"
  echo "  paper_draft_v5.tex -> Drive root"
  echo "  Consolidation_v4   -> Drive root"
  echo "  Context_Update_188 -> Drive root"

  # Copy paper_experiments outputs (the bootstrap summaries and decision)
  mkdir -p "$DRIVE_DIR/paper_experiments_outputs_20260630"
  for f in \
    paper_experiments/outputs/bow_experiments_summary.csv \
    paper_experiments/outputs/bow_experiments_summary.md \
    paper_experiments/outputs/bow_experiments_decision.txt \
    paper_experiments/outputs/power_analysis.md
  do
    if [ -f "$f" ]; then
      cp "$f" "$DRIVE_DIR/paper_experiments_outputs_20260630/"
      echo "  $(basename "$f")"
    fi
  done

  # Bundle the raw .txt outputs as a zip to avoid clogging Drive root
  if command -v zip >/dev/null 2>&1; then
    raw_zip="$DRIVE_DIR/paper_experiments_outputs_20260630/raw_outputs.zip"
    rm -f "$raw_zip"
    zip -q -j "$raw_zip" paper_experiments/outputs/bow_statins_*.txt 2>/dev/null
    echo "  raw_outputs.zip (raw .txt outputs)"
  fi

  echo ""
  echo "Drive sync complete."
fi
echo ""

# ---------- Summary ----------
echo "============================================================"
echo "Storage complete"
echo "============================================================"
echo "Local:   committed to $REPO_ROOT, SHA $commit_sha"
if [ $push_status -eq 0 ]; then
  echo "GitHub:  pushed to https://github.com/SamInMotion/Medical-intervention-text-classification"
else
  echo "GitHub:  push pending (commit local only)"
fi
if [ $SKIP_DRIVE -eq 0 ]; then
  echo "Drive:   synced to $DRIVE_DIR"
fi
echo ""
echo "Next session can re-enter context by reading, in order:"
echo "  1. docs/Cohen_BERT_Extension_Results_Consolidation_v4.md (paper truth)"
echo "  2. docs/Context_Update_188.md (session closeout, next actions)"
echo "  3. paper/paper_draft_v5.tex (current manuscript)"
