#!/usr/bin/env bash
# step1_docs.sh — work the documentation cleanup before commit.
#
#     bash scripts/step1_docs.sh report     # every outstanding item, with lines
#     bash scripts/step1_docs.sh manifest   # regenerate MANIFEST.md
#     bash scripts/step1_docs.sh precommit  # gate before committing
#
# This script does NOT edit prose. Each defect is shown as file:line:text so the
# edit is made against the real string. Four Check-3 entries are NOT defects and
# are marked as such; editing them would delete working documentation.
#
# Run `report`, fix, run `report` again, repeat until clean, then `precommit`.

set -u
cd "$(git rev-parse --show-toplevel)" 2>/dev/null || { echo "not a git repo" >&2; exit 2; }
PY=python; command -v python >/dev/null 2>&1 || PY=python3
CMD="${1:-report}"

hdr()  { printf '\n\033[1m=== %s\033[0m\n' "$1"; }
item() { printf '\n  [%s] %s\n' "$1" "$2"; }
act()  { printf '      -> %s\n' "$1"; }
show() { if [ -n "$1" ]; then printf '%s\n' "$1" | sed 's/^/        /'; else printf '        (clear)\n'; fi; }

scan() {  # scan <id> <what> <action> <regex>
  local id="$1" what="$2" action="$3" pat="$4" hits
  hits=$(git grep -n -E "$pat" -- '*.md' '*.py' '*.sh' 2>/dev/null \
         | grep -v '^docs/CORRECTIONS_v7_7' \
         | grep -v '^docs/PROVENANCE_ADDENDUM' \
         | grep -v '^archive/' | grep -v '^paper/' \
         | grep -v '^scripts/fig_design_sensitivity.py' \
         | grep -v '^paper_experiments/power_analysis.py')
  item "$id" "$what"
  act "$action"
  show "$hits"
  [ -n "$hits" ] && return 1 || return 0
}

if [ "$CMD" = "manifest" ]; then
  hdr "regenerating MANIFEST.md"
  if [ -f scripts/build_manifest.py ]; then
    $PY scripts/build_manifest.py && echo "ok  MANIFEST.md rebuilt"
  elif [ -f build_manifest.py ]; then
    $PY build_manifest.py && echo "ok  MANIFEST.md rebuilt"
  else
    echo "FAIL  build_manifest.py not found on this branch." >&2
    echo "      CU 213 records it as anon-submission only. Port it to main first." >&2
    exit 1
  fi
  exit 0
fi

if [ "$CMD" = "precommit" ]; then
  fail=0
  hdr "1. working tree"
  git status --short
  hdr "2. documentation defects"
  bash "$0" report >/tmp/_r1 2>&1
  n=$(grep -c '^        [^(]' /tmp/_r1 || true)
  [ "$n" -eq 0 ] && echo "ok    no outstanding defect lines" \
                 || { echo "FAIL  $n defect line(s) remain — run: bash $0 report"; fail=1; }
  rm -f /tmp/_r1
  hdr "3. branch checks 3 and 5"
  if [ -f scripts/verify_branch.sh ]; then
    bash scripts/verify_branch.sh >/tmp/_vb 2>&1
    for s in "3. documentation paths" "5. orphans"; do
      if awk -v s="$s" '/^=== /{cur=$0} /^FAIL/{if (index(cur,s)) print}' /tmp/_vb | grep -q .; then
        echo "FAIL  branch check $s still failing"; fail=1
      else echo "ok    branch check $s"; fi
    done
    rm -f /tmp/_vb
  else echo "skip  verify_branch.sh absent"; fi
  hdr "4. v7.7 repository state"
  bash scripts/verify_v7_7.sh >/tmp/_v7 2>&1
  grep -E '^(FAIL|ok    ALL)' /tmp/_v7 | head -12 | sed 's/^/  /'
  rm -f /tmp/_v7
  hdr "5. remote"
  ahead=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo '?')
  echo "  main is $ahead commit(s) ahead of origin"
  printf '\n-----\n'
  [ "$fail" -eq 0 ] && echo "READY TO COMMIT" || echo "NOT READY — see FAIL lines"
  exit "$fail"
fi

# ---------------------------------------------------------------- report
cat <<'INTRO'

STEP 1 WORKLIST — documentation cleanup before commit
Each item shows file:line:text. Edit against those strings, not from memory.
Re-run this command after each fix.
INTRO

hdr "A. Superseded values and claims"

scan A1 "stale BERT interval / seed-level values" \
  "Superseded by the per-fold set (ledger #20). PROVENANCE rows recording the correction are legitimate; anything else is a live carrier." \
  '\-0\.011|\+0\.052|0\.0022|0\.0733'

scan A2 "'order of magnitude'" \
  "The factor is about five. Origin: Consolidation v4 section 3 claim 3." \
  'order of magnitude'

scan A3 "t-correction understated as under 4 percent" \
  "It is +33% at n=5. The 3.0% figure is the n=35 value. This is the origin of the n=5/n=35 conflation." \
  '4\s*%\s*at\s*n\s*=\s*5|<\s*4\s*%'

scan A4 "effect attributed to the literature" \
  "Traces to this work's own April and June runs (ledger #23), not to any published value." \
  "literature'?s? reported|reported by prior work|replicating prior work|literature has reported"

scan A5 "superseded MDE values" \
  "Replaced by exact noncentral-t 0.114 / 0.254 / 0.384. CHECK EACH: some hits are legitimate audit records of the correction." \
  '0\.085|0\.189|0\.286'

scan A6 "old Table 9 SD column" \
  "Corrected to 0.064 / 0.171 / 0.162 from outputs/bow_*_multirun_summary.json." \
  'SD.*0\.067|0\.170.*0\.160'

scan A7 "'design-limited' claim" \
  "v7.7 scopes this to the single-run design. Context updates carrying it unscoped need the same scoping." \
  'design-limited'

scan A8 "superseded X1 instruction" \
  "X1 named make_fig2_design_sensitivity.py, which does not produce the consumed figure and is now archived. Ledger #32 is CLOSED: scripts/fig_design_sensitivity.py is the generator." \
  'make_fig2_design_sensitivity'

scan A9 "'Audit closed; no blocking unknowns remain'" \
  "Written over three OPEN rows. Correct the changelog line." \
  'Audit closed'

hdr "B. Branch check 3 — documentation paths"
cat <<'B'

  Run for the live list:
      bash scripts/verify_branch.sh 2>&1 | sed -n '/3. documentation paths/,/=== 4/p'

  NOT DEFECTS — do not edit these four:
      README.md:107        abstracts.tsv / neo.json / med-stopwords.txt
      data/README.md:107   same
        These are SETUP INSTRUCTIONS for user-supplied thesis files. Removing
        them deletes working documentation. (They also match the FINGERPRINT
        pattern, so if the checker must pass on anon-submission, scope the
        pattern rather than the prose.)
      REPRODUCING.md:93    "..._smoke_rerun2.txt"
        Prose ellipsis, not a broken link. MANIFEST.md:65 records the file as
        present and cited.

  GENUINE — fix these:
      make_fig1_gap_forest_v3.py  x3   README.md:72,163  data/README.md:72,163
        No such file. Script is scripts/make_fig1_gap_forest.py; _v3 is its
        OUTPUT. Ledger #39 already records this. See also scripts/README.md:11.
      bootstrap_bert_per_fold.py  x2   data/README.md:71,160
        Restored from 992d227^ to scripts/. Confirm the path in both lines.
      med-stopwords.txt           x2
        Genuinely missing. Either ship it or drop the reference.
      docs/Context_Update_188.md       PROVENANCE.md
        Ledger #42. Context Updates are project documents, not repo files.
        Cite as "CU 188" without the .md so the checker stops parsing it as a path.
B

hdr "C. Branch check 5 — orphans"
cat <<'C'

  Run for the live list:
      bash scripts/verify_branch.sh 2>&1 | sed -n '/5. orphans/,$p'

  DO NOT ARCHIVE — reference these in PROVENANCE.md instead:
      scripts/run_bow_multirun.sh
        Produced the seven reruns behind every BoW table. Provenance evidence.
      paper_experiments/outputs/run_statins_{10fold,subsampling}_*.log   x8
        Run artifacts for the subsampling and 10-fold analyses (30 June).

  SAFE TO ARCHIVE:
      Main Classify Abstracts Code.ipynb, Ontology Preferred Label Groupings.ipynb
      outputs/archive/fig1_gap_forest_v2.{pdf,png}   (belong with archive/make_fig1_v2.py)
      paper_experiments/commit_and_store.sh, paper_experiments/verify_setup.sh

  ORPHAN, NEEDS A MANIFEST ENTRY, DO NOT DELETE:
      docs/Cohen_BERT_Extension_Results_Consolidation_v4.md
        Same file carrying A1, A2 and A3 above.

  WILL BECOME ORPHANS ON COMMIT — add manifest entries in the same commit:
      docs/CORRECTIONS_v7_7.md, docs/PROVENANCE_ADDENDUM_v7_7.md,
      scripts/apply_v7_7_repo.sh, scripts/verify_v7_7.sh, scripts/step1_docs.sh,
      scripts/bootstrap_bert_per_fold.py, archive/pre_v7_7/
C

hdr "D. PROVENANCE.md merge"
cat <<'D'

  From docs/PROVENANCE_ADDENDUM_v7_7.md:
      Section A  corrections to rows #31 (CLOSED), #35 (misdirected), #12, #13, #22
      Section B  new rows B1-B5: Table 9 SDs, exact MDEs, n=35 MDEs,
                 benchmark label counts, Nadeau-Bengio implementation defect
      Section F  repository findings F1-F8
  Sections C, D, E are worklists, not ledger rows — leave them in the addendum.

  Also fold in: Rule 1 backfill (addendum C) for the numbers currently in the
  manuscript with no row.
D

hdr "E. Then"
cat <<'E'
  bash scripts/step1_docs.sh manifest
  bash scripts/step1_docs.sh precommit
  git add -p     # read the diff
  git commit
  git push origin main
E
echo
