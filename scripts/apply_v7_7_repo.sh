#!/usr/bin/env bash
# apply_v7_7_repo.sh — the remaining v7.7 repository work.
#
#     bash scripts/apply_v7_7_repo.sh [staging-dir]
#
# Assumes you have ALREADY done, manually:
#   - paper_experiments/power_analysis.py replaced with the patched version
#   - outputs/fig_design_sensitivity_final.pdf regenerated
#   - paper/paper_draft_v7_7_nejlt.tex copied in and compiled
#
# The optional staging-dir is where the downloaded fig_design_sensitivity.py
# lives. Without it the script skips that step and says so.
#
# This script does not commit, does not touch anon-submission, and does not
# rewrite prose. Documentation defects are DETECTED and written to
# docs/CORRECTIONS_v7_7.md with file, line and text, so the edits are made
# with the real strings in front of you rather than by a regex written blind.
#
# Everything it replaces is copied to archive/pre_v7_7/ first.

set -u
STAGE="${1:-}"
cd "$(git rev-parse --show-toplevel)" 2>/dev/null || { echo "not a git repo" >&2; exit 2; }

PY=python; command -v python >/dev/null 2>&1 || PY=python3
BK=archive/pre_v7_7
mkdir -p "$BK" archive docs

fail=0
note() { printf '\n=== %s\n' "$1"; }
bad()  { printf 'FAIL  %s\n' "$1"; fail=1; }
ok()   { printf 'ok    %s\n' "$1"; }
skip() { printf 'skip  %s\n' "$1"; }

echo "repo:   $(pwd)"
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
echo "python: $($PY --version 2>&1)"

# ---------------------------------------------------------------- 0. preflight
note "0. what is already done"
grep -q 'multirun_diffs_for_topic' paper_experiments/power_analysis.py 2>/dev/null \
  && ok "power_analysis.py is the patched version" \
  || bad "power_analysis.py is NOT patched — replace it before continuing"
[ -f outputs/fig_design_sensitivity_final.pdf ] \
  && ok "fig_design_sensitivity_final.pdf present ($(date -r outputs/fig_design_sensitivity_final.pdf '+%Y-%m-%d %H:%M' 2>/dev/null))" \
  || bad "outputs/fig_design_sensitivity_final.pdf missing"
ls paper/paper_draft_v7_7*.tex >/dev/null 2>&1 \
  && ok "v7.7 manuscript in paper/" || skip "no paper/paper_draft_v7_7*.tex (fine if kept elsewhere)"

# ------------------------------------------------- 1. patched figure generator
note "1. figure generator"
if grep -q 'N_BOW' scripts/fig_design_sensitivity.py 2>/dev/null; then
  ok "scripts/fig_design_sensitivity.py already patched"
elif [ -n "$STAGE" ] && [ -f "$STAGE/fig_design_sensitivity.py" ]; then
  cp scripts/fig_design_sensitivity.py "$BK/" 2>/dev/null
  cp "$STAGE/fig_design_sensitivity.py" scripts/fig_design_sensitivity.py
  ok "installed patched scripts/fig_design_sensitivity.py (old copy in $BK)"
else
  bad "scripts/fig_design_sensitivity.py is the old version and no staging copy given"
  echo "      the shipped PDF is already correct, but the generator that made it is not"
fi

# --------------------------------------------- 2. retire superseded generators
note "2. retire superseded figure generators"
for f in scripts/make_fig1_v2.py make_fig2_design_sensitivity.py; do
  if [ -f "$f" ]; then
    mkdir -p archive
    git mv -f "$f" "archive/$(basename "$f")" 2>/dev/null || mv -f "$f" "archive/$(basename "$f")"
    ok "archived $f"
  else
    skip "$f not present"
  fi
done

# --------------------------------------------------- 3. label the decoy fig1
note "3. label the remaining fig1 generator"
F=scripts/make_fig1_gap_forest.py
if [ ! -f "$F" ]; then
  skip "$F not present"
elif grep -q 'NOT the generator behind Figure 1' "$F"; then
  ok "$F already labelled"
else
  cp "$F" "$BK/$(basename "$F")"
  {
    echo '# NOTE: writes fig1_gap_forest_v3.pdf, which the manuscript does NOT include.'
    echo '# The shipped outputs/fig1_gap_forest.pdf comes from scripts/make_paper_artifacts.py.'
    echo '# This is NOT the generator behind Figure 1.'
    cat "$F"
  } > "$F.tmp" && mv "$F.tmp" "$F"
  ok "header added to $F"
fi

# ------------------------------------------------------ 4. regenerate outputs
note "4. regenerate power analysis"
if $PY paper_experiments/power_analysis.py > /tmp/_pa 2>&1; then
  ok "power_analysis.py ran"
  grep -E '^\s+(Statins|Opiods|ADHD)' /tmp/_pa | sed 's/^/      /'
else
  bad "power_analysis.py failed:"; sed 's/^/      /' /tmp/_pa | head -8
fi
rm -f /tmp/_pa

# -------------------------------------------------- 5. detect documentation debt
note "5. scan for documentation defects"
OUT=docs/CORRECTIONS_v7_7.md
{
  echo "# Documentation corrections owed by v7.7"
  echo
  echo "Generated $(date '+%Y-%m-%d %H:%M') by scripts/apply_v7_7_repo.sh."
  echo "Detected, not fixed. Each entry is file:line and the text as it stands."
  echo
} > "$OUT"

scan() {  # scan <label> <extended-regex> <what to do>
  local label="$1" pat="$2" action="$3" hits
  # paper/*.tex is excluded: its header change-log quotes every corrected
  # phrase as a record of the correction. The manuscript body is checked
  # separately, with comments stripped, by scripts/verify_v7_7.sh check 6.
  hits=$(git grep -n -E "$pat" -- '*.md' '*.py' '*.sh' 2>/dev/null \
         | grep -v '^docs/CORRECTIONS_v7_7.md' \
         | grep -v '^docs/PROVENANCE_ADDENDUM' \
         | grep -v '^archive/')
  {
    echo "## $label"
    echo
    echo "$action"
    echo
    if [ -n "$hits" ]; then echo '```'; echo "$hits"; echo '```'; else echo "_none found_"; fi
    echo
  } >> "$OUT"
  if [ -n "$hits" ]; then printf 'found %-38s %s occurrence(s)\n' "$label" "$(printf '%s\n' "$hits" | wc -l | tr -d ' ')"
  else printf 'clean %s\n' "$label"; fi
}

scan "stale BERT interval / seed-level values" \
     '\-0\.011|\+0\.052|0\.0022|0\.0733' \
     "Superseded by the per-fold set (ledger #20). PROVENANCE.md rows that record the correction are legitimate; anything else is a live carrier."

scan "'order of magnitude'" \
     'order of magnitude' \
     "The factor is about five, not ten. Originates in Consolidation v4 §3 claim 3."

scan "t-correction understated as under 4 percent" \
     '4\s*%\s*at\s*n\s*=\s*5|<\s*4\s*%|4% relative to the exact' \
     "It is +33% at n=5. The 3.0% figure is the value at n=35, so the check that licensed the normal approximation was run at the wrong fold count."

scan "effect attributed to the literature" \
     "literature'?s? reported|reported by prior work|replicating prior work|literature has reported" \
     "Traces to this work's own April and June runs (ledger #23), not to any published value."

scan "superseded MDE values" \
     '0\.085|0\.189|0\.286' \
     "Replaced by the exact noncentral-t values 0.114 / 0.254 / 0.384. Check each hit: some are legitimate audit records."

scan "old Table 9 SD column" \
     'SD.*0\.067|0\.170.*0\.160|0\.067.*0\.170' \
     "Corrected to 0.064 / 0.171 / 0.162 from outputs/bow_*_multirun_summary.json."

scan "'design-limited' claim" \
     'design-limited' \
     "Scoped in v7.7 to the single-run design. Context updates carrying the unscoped claim need the same scoping."

scan "superseded X1 instruction" \
     'make_fig2_design_sensitivity' \
     "X1 told you to edit and regenerate this file. The consumed figure comes from scripts/fig_design_sensitivity.py; the named file is superseded and now archived."

scan "'Audit closed; no blocking unknowns remain'" \
     'Audit closed' \
     "Written over three OPEN rows. Correct the changelog line."

{
  echo "## Not detectable by grep"
  echo
  echo "- \`REPRODUCING.md\` Table 9 entry marks the whole table **V** while verifying only the two means from the long CSV. It never checked the SD column or the three multi-run rows. State what it covers."
  echo "- \`scripts/verify_branch.sh\` \`IDENTITY\` implements identity tokens plus the machine name, but its header claims to cover \"local path\". A path with no name in it passes. Consider adding a path shape and an archive-identifier pattern (\`10\\.5281/zenodo\`, \`arxiv\\.org\`)."
  echo "- \`PROVENANCE.md\` needs sections A, B and F of \`docs/PROVENANCE_ADDENDUM_v7_7.md\` merged in."
} >> "$OUT"
ok "wrote $OUT"

# ------------------------------------------------------------- 6. branch state
note "6. branch state"
git fetch --prune --quiet 2>/dev/null && ok "fetched and pruned" || skip "fetch failed (offline?)"
if git show-ref --verify --quiet refs/remotes/origin/v2.0-infastructure; then
  bad "origin/v2.0-infastructure EXISTS — public unaudited surface, handover item stands"
else
  ok "no origin/v2.0-infastructure — local only, downgrade the handover item"
fi
ahead=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo '?')
[ "$ahead" = "0" ] && ok "level with origin/main" || echo "note  main is $ahead commit(s) ahead of origin"

# ------------------------------------------------------------------- 7. status
note "7. working tree"
git status --short

cat <<MANUAL

-----
NEXT
  1  work through docs/CORRECTIONS_v7_7.md
  2  merge docs/PROVENANCE_ADDENDUM_v7_7.md sections A, B, F into PROVENANCE.md
  3  bash scripts/verify_v7_7.sh
  4  git add -p, read the diff, commit, push main
  5  anon branch and mirror at submission time, not now
MANUAL

exit "$fail"
