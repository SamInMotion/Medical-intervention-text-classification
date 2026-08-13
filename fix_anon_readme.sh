#!/usr/bin/env bash
# fix_anon_readme.sh — correct the four false statements in the anon-submission README.
#
#     bash fix_anon_readme.sh          # dry run, shows the diff
#     bash fix_anon_readme.sh --write
#
# Must be run on anon-submission. main has its own README and is not touched.
#
# What is wrong and why:
#   1  Figures block names two commands that do not produce the shipped figures.
#      make_paper_artifacts.py crashes on the current tree (ledger #43: expects
#      bert_{topic}_{mode}.txt without the seed suffix). make_fig2_design_sensitivity.py
#      is archived and writes a file nothing consumes (#32, #35).
#   2  Notes paragraph inverts which fig1 script is authoritative (#53b) and
#      names scripts/make_fig1_v2.py, which is archived (#53).
#   3  Layout and the paragraph below claim REPRODUCING.md is generated from
#      PROVENANCE.md. It is hand-maintained; #45-#48 record it as wrong about
#      its own repository.
#   4  parse_bow_multirun.py is cited at repo root; #46 records it as moved.
#
# Backup written to README.md.readme.bak -- remove before commit (ledger #49).

set -u
cd "$(git rev-parse --show-toplevel)" 2>/dev/null || { echo "not a git repo" >&2; exit 2; }
BR=$(git rev-parse --abbrev-ref HEAD)
[ "$BR" = "anon-submission" ] || { echo "FAIL  on '$BR', expected 'anon-submission'" >&2; exit 1; }
[ -f README.md ] || { echo "FAIL  README.md not found" >&2; exit 1; }

WRITE=0
[ "${1:-}" = "--write" ] && WRITE=1

TMP=$(mktemp)
cp README.md "$TMP"

python - "$TMP" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
hits, misses = [], []

def sub(label, old, new):
    global s
    if old in s:
        s = s.replace(old, new)
        hits.append(label)
    else:
        misses.append(label)

# 1. Figures block
sub("figures block",
"""python scripts/make_fig1_gap_forest.py       # reads the six summary JSONs
python make_fig2_design_sensitivity.py       # values hardcoded, see PROVENANCE.md""",
"""python scripts/fig_design_sensitivity.py     # Figure 2; values hardcoded, see PROVENANCE.md""")

# 1b. paragraph after the figures block
sub("figures note",
"Notes\n",
"""Figure 1 (`outputs/fig1_gap_forest.pdf`) is archived rather than regenerable at
this commit. `scripts/make_paper_artifacts.py` names it but expects per-fold files
without the seed suffix the multi-seed protocol introduced, so it fails on the
current tree (`PROVENANCE.md` #43). `scripts/make_fig1_gap_forest.py` runs, but
writes `fig1_gap_forest_v3.pdf`, which the manuscript does not include. The
archived per-fold outputs under `outputs/` reproduce every reported value.

Notes
""")

# 2. Notes paragraph on the fig1 scripts
sub("notes paragraph",
"`scripts/make_fig1_v2.py` and `scripts/make_paper_artifacts.py` predate the current output naming and are retained as record, not as regeneration paths.",
"`scripts/make_fig1_gap_forest.py` writes `fig1_gap_forest_v3.pdf`, which the manuscript does not include; it is retained as record. `scripts/make_paper_artifacts.py` names the shipped `outputs/fig1_gap_forest.pdf` but predates the multi-seed output naming and does not run on the current tree; see `PROVENANCE.md` #43 and #53b.")

# 3. REPRODUCING.md provenance claim, two sites
sub("layout line",
"REPRODUCING.md           claim-to-command map, generated from PROVENANCE.md",
"REPRODUCING.md           companion claim-to-command map")

sub("authoritative paragraph",
"`REPRODUCING.md` is generated from it; where the two disagree, `PROVENANCE.md` wins.",
"`REPRODUCING.md` is a hand-maintained companion; where the two disagree, `PROVENANCE.md` wins, and `PROVENANCE.md` #45-#48 record where `REPRODUCING.md` is known to be stale.")

open(p, "w", encoding="utf-8").write(s)
print("APPLIED")
for h in hits:
    print("  ok   " + h)
if misses:
    print("NOT APPLIED -- edit by hand")
    for m in misses:
        print("  MISS " + m)
PY

echo
echo "=== parse_bow_multirun.py path check (ledger #46) ==="
ACT=$(git ls-files | grep -i 'parse_bow_multirun' | head -1)
if [ -z "$ACT" ]; then
  echo "  not tracked on this branch -- remove the command from README or restore the file"
elif grep -q "python parse_bow_multirun.py" "$TMP"; then
  if [ "$ACT" = "parse_bow_multirun.py" ]; then
    echo "  ok   cited path matches tracked path"
  else
    sed -i "s|python parse_bow_multirun.py|python $ACT|" "$TMP"
    echo "  fixed  parse_bow_multirun.py -> $ACT"
  fi
else
  echo "  no citation found in README"
fi

echo
if diff -q README.md "$TMP" >/dev/null; then
  echo "no change"; rm -f "$TMP"; exit 0
fi

if [ "$WRITE" -eq 0 ]; then
  echo "=== diff (dry run; rerun with --write) ==="
  diff -u README.md "$TMP" | sed -n '3,200p'
  rm -f "$TMP"
  exit 0
fi

cp README.md README.md.readme.bak
mv "$TMP" README.md
echo "wrote README.md (backup README.md.readme.bak)"
cat <<'NEXT'

next:
  rm -f README.md.readme.bak            # ledger #49: no tracked .bak
  bash scripts/verify_branch.sh
  git add -A && git commit -m "README: correct the figure regeneration commands and REPRODUCING provenance claim"
  git push origin anon-submission
  # then re-sync the 4open.science mirror -- auto-update is off, it is pinned to ae06894

still yours, in PROVENANCE.md:
  #43   add make_paper_artifacts.py alongside bootstrap_paired_permutation.py;
        both fail for the same seed-suffix reason
  #53b  add that Figure 1's generator is identified but not executable at this
        commit -- git grep savefig proved the filename, not that it runs
NEXT
