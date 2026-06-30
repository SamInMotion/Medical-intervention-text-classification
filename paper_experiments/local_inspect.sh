#!/usr/bin/env bash
# ============================================================================
# local_inspect.sh
# ----------------------------------------------------------------------------
# Inspect the LOCAL working tree. Run from inside the repo:
#   cd /c/Users/samue.KITAB/Medical-intervention-text-classification
#   bash paper_experiments/local_inspect.sh
#
# Read-only. Dumps three files in full and shows git state.
# ============================================================================

set -u

OUT="local_inspect_$(date +%Y%m%d_%H%M%S).txt"
: > "$OUT"

section() {
  printf '\n========================================================================\n'  >> "$OUT"
  printf '== %s\n'                                                                       "$1" >> "$OUT"
  printf '========================================================================\n'    >> "$OUT"
}

section "1. PYTHON / VENV STATUS"
echo "which python:" >> "$OUT"
which python        >> "$OUT" 2>&1
echo ""             >> "$OUT"
echo "python -c 'import sys; print(sys.executable); print(sys.version)':" >> "$OUT"
python -c "import sys; print(sys.executable); print(sys.version)" >> "$OUT" 2>&1
echo ""             >> "$OUT"
echo "VIRTUAL_ENV env var: ${VIRTUAL_ENV:-<unset>}" >> "$OUT"
echo ""             >> "$OUT"
echo "Looking for .venv* directories at repo root:" >> "$OUT"
ls -d .venv* 2>/dev/null >> "$OUT" || echo "  none found" >> "$OUT"
echo ""             >> "$OUT"
echo "Try: ./.venv312/Scripts/python --version" >> "$OUT"
./.venv312/Scripts/python --version 2>&1 >> "$OUT" || true
echo ""             >> "$OUT"
echo "Try: ./.venv312/Scripts/python -c 'import tensorflow; print(tensorflow.__version__)'" >> "$OUT"
./.venv312/Scripts/python -c "import tensorflow; print(tensorflow.__version__)" 2>&1 >> "$OUT" || true

section "2. GIT STATE"
git rev-parse --abbrev-ref HEAD >> "$OUT" 2>&1
echo "" >> "$OUT"
git status                       >> "$OUT" 2>&1
echo "" >> "$OUT"
echo "--- git log -10 --oneline ---" >> "$OUT"
git log -10 --oneline            >> "$OUT" 2>&1
echo "" >> "$OUT"
echo "--- git stash list ---" >> "$OUT"
git stash list                   >> "$OUT" 2>&1

section "3. LOCAL FILES — full dump"
for f in src/features.py src/preprocessing.py; do
  echo "" >> "$OUT"
  echo ">>>>> BEGIN FILE: $f <<<<<" >> "$OUT"
  if [ -f "$f" ]; then
    cat "$f" >> "$OUT"
  else
    echo "[FILE NOT FOUND]" >> "$OUT"
  fi
  echo "" >> "$OUT"
  echo "<<<<< END FILE: $f >>>>>" >> "$OUT"
done

section "4. LOCAL cohen_pipeline.py — only the subsample-related blocks"
echo "" >> "$OUT"
if [ -f "src/cohen_pipeline.py" ]; then
  echo "--- grep -nC 5 'subsample' src/cohen_pipeline.py ---" >> "$OUT"
  grep -nC 5 -i "subsample" src/cohen_pipeline.py >> "$OUT" 2>&1 || echo "[no matches]" >> "$OUT"
  echo "" >> "$OUT"
  echo "--- argparse block (looking for --subsample) ---" >> "$OUT"
  grep -nC 3 -- "--subsample" src/cohen_pipeline.py >> "$OUT" 2>&1 || echo "[no matches]" >> "$OUT"
else
  echo "[src/cohen_pipeline.py NOT FOUND]" >> "$OUT"
fi

section "5. DIFFS — local vs origin/main"
git fetch origin --quiet 2>/dev/null || true
for f in src/features.py src/preprocessing.py src/cohen_pipeline.py; do
  echo "" >> "$OUT"
  echo "--- git diff origin/main -- $f ---" >> "$OUT"
  git diff origin/main -- "$f" >> "$OUT" 2>&1
done

section "6. ANY .bak FILES IN src/"
ls -la src/*.bak 2>/dev/null >> "$OUT" || echo "[no .bak files]" >> "$OUT"

section "7. SHA256 OF LOCAL FILES"
for f in src/features.py src/preprocessing.py src/cohen_pipeline.py; do
  if [ -f "$f" ]; then
    printf '  %s  %s\n' "$(sha256sum "$f" | awk '{print $1}')" "$f" >> "$OUT"
  fi
done

echo ""
echo "Report saved to: $OUT"
echo "Lines: $(wc -l < "$OUT")    Bytes: $(wc -c < "$OUT")"
echo "Paste the contents back."
