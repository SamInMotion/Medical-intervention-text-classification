#!/usr/bin/env bash
# Pre-mirror verification for the anonymous review branch.
#
# Run before generating or regenerating the Anonymous GitHub mirror:
#     bash scripts/verify_branch.sh
#
# Five checks. Any failure exits non-zero and should block the mirror.
#   1  identity      no name, institution, handle, or local path in tracked files
#   2  fingerprint   no provenance detail that resolves to the source thesis
#   3  paths         every file the documentation cites actually exists
#   4  code          the analysis modules import and the test suite passes
#   5  orphans       every tracked file maps to an import or a document
#
# Check 3 is what catches documentation drifting away from the tree, which is
# how REPRODUCING.md came to name a file that had been renamed.

set -u
cd "$(dirname "$0")/.." || exit 2
fail=0
note() { printf '\n=== %s\n' "$1"; }
bad()  { printf 'FAIL  %s\n' "$1"; fail=1; }
ok()   { printf 'ok    %s\n' "$1"; }

IDENTITY='okoe|okmens|samuel|bergen|SamInMotion|KITAB'
FINGERPRINT='dementia|NEO ontology|neo\.json|SNOMED|1,611|abstracts\.tsv'

note "1. identity"
hits=$(git ls-files -z | xargs -0 grep -l -i -E "$IDENTITY" 2>/dev/null \
       | grep -v '^data/cohen/' | grep -v '^scripts/verify_branch.sh$')
if [ -n "$hits" ]; then bad "identifying strings:"; echo "$hits" | sed 's/^/      /'
else ok "no identifying strings in tracked files"; fi

note "2. thesis fingerprint"
hits=$(git ls-files -z | xargs -0 grep -l -i -E "$FINGERPRINT" 2>/dev/null \
       | grep -v '^data/cohen/' | grep -v '^scripts/verify_branch.sh$')
if [ -n "$hits" ]; then bad "provenance detail:"; echo "$hits" | sed 's/^/      /'
else ok "no thesis fingerprint in tracked files"; fi

note "3. documentation paths"
missing=0
for doc in PROVENANCE.md REPRODUCING.md README.md data/README.md; do
  [ -f "$doc" ] || continue
  grep -o '`[^`]*`' "$doc" \
    | tr -d '`' \
    | grep -E '\.(py|sh|json|csv|tsv|ipynb|md|toml|txt)$' \
    | grep -v '^\*' | grep -v '[{}]' | sort -u \
    | while read -r p; do
        [ -e "$p" ] || [ -n "$(git ls-files "*$(basename "$p")" | head -1)" ] \
          || echo "      $doc -> $p"
      done
done > /tmp/_missing_paths 2>/dev/null
if [ -s /tmp/_missing_paths ]; then
  bad "documentation cites paths that do not exist:"; cat /tmp/_missing_paths
else ok "every cited path exists"; fi
rm -f /tmp/_missing_paths

note "4. code"
if python -c "
from src import cohen_pipeline, auto_mesh, benchmark_loader, evaluation
from src import features, preprocessing, config, models
" 2>/dev/null; then ok "analysis modules import"
else bad "analysis modules do not import (is .venv312 active?)"; fi
# cohen_bert_pipeline needs torch and runs on Colab; not checked here.

if python -m pytest tests/ -q >/tmp/_pytest 2>&1; then
  ok "$(tail -1 /tmp/_pytest)"
else bad "test suite: $(tail -1 /tmp/_pytest)"; fi
rm -f /tmp/_pytest

note "5. orphans"
if python scripts/build_manifest.py; then ok "no orphans"
else bad "orphaned files, see MANIFEST.md"; fi

note "result"
if [ "$fail" -eq 0 ]; then
  echo "PASS  branch is ready to mirror"
else
  echo "FAIL  do not generate the mirror until the above are resolved"
fi
exit "$fail"
