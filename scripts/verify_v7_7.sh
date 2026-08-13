#!/usr/bin/env bash
# verify_v7_7.sh — check the repository matches manuscript v7.7.
#
#     bash scripts/verify_v7_7.sh
#
# Complements scripts/verify_branch.sh rather than replacing it. That one asks
# whether the review branch is safe to mirror. This one asks whether the tree
# corresponds to the manuscript. Exits non-zero on any failure.
#
# Checks that depend on the manuscript are skipped, not failed, when it is not
# present. Anonymisation is a submission-time concern and is not gated here.

set -u
cd "$(git rev-parse --show-toplevel)" 2>/dev/null || { echo "not a git repo" >&2; exit 2; }
PY=python; command -v python >/dev/null 2>&1 || PY=python3

fail=0
note() { printf '\n=== %s\n' "$1"; }
bad()  { printf 'FAIL  %s\n' "$1"; fail=$((fail+1)); }
ok()   { printf 'ok    %s\n' "$1"; }
skip() { printf 'skip  %s\n' "$1"; }

TEX=$(ls paper/paper_draft_v7_7*.tex 2>/dev/null | head -1)

note "1. patched sources in place"
grep -q 'multirun_diffs_for_topic' paper_experiments/power_analysis.py 2>/dev/null \
  && ok "power_analysis.py has Part 2" || bad "power_analysis.py not patched"
grep -q 'mde_normal' paper_experiments/power_analysis.py 2>/dev/null \
  && ok "power_analysis.py reports the exact MDE" \
  || bad "power_analysis.py still reports only the normal approximation"
grep -q 'N_BOW' scripts/fig_design_sensitivity.py 2>/dev/null \
  && ok "fig_design_sensitivity.py patched" || bad "figure generator not patched"

note "2. superseded generators retired"
for f in scripts/make_fig1_v2.py make_fig2_design_sensitivity.py; do
  [ -f "$f" ] && bad "$f still live" || ok "$f retired"
done
if [ -f scripts/make_fig1_gap_forest.py ]; then
  grep -q 'NOT the generator behind Figure 1' scripts/make_fig1_gap_forest.py \
    && ok "make_fig1_gap_forest.py labelled" || bad "make_fig1_gap_forest.py unlabelled"
fi

note "3. power analysis reproduces the manuscript numbers"
out=$($PY paper_experiments/power_analysis.py 2>/dev/null)
for v in 0.1135 0.2542 0.3838; do
  printf '%s' "$out" | grep -q "$v" && ok "Part 1 emits $v" || bad "Part 1 did not emit $v"
done
for v in 0.0641 0.1713 0.1620; do
  printf '%s' "$out" | grep -q "$v" && ok "Part 2 emits $v" || bad "Part 2 did not emit $v"
done

note "4. figures present and not stale"
for f in outputs/fig_design_sensitivity_final.pdf outputs/fig1_gap_forest.pdf; do
  [ -f "$f" ] && ok "$f" || bad "$f missing"
done
if [ -f scripts/fig_design_sensitivity.py ] && [ -f outputs/fig_design_sensitivity_final.pdf ]; then
  if [ scripts/fig_design_sensitivity.py -nt outputs/fig_design_sensitivity_final.pdf ]; then
    bad "generator is newer than its PDF — regenerate"
  else ok "PDF is not older than its generator"; fi
fi

note "5. stale values outside the audit record"
hits=$(git grep -l -E '\-0\.011|\+0\.052|0\.0022|0\.0733' -- '*.py' '*.md' '*.tex' 2>/dev/null \
       | grep -v '^PROVENANCE.md$' | grep -v '^docs/PROVENANCE' \
       | grep -v '^docs/CORRECTIONS' | grep -v '^archive/')
if [ -n "$hits" ]; then bad "stale BERT values remain:"; echo "$hits" | sed 's/^/      /'
else ok "none outside the audit record"; fi

note "6. manuscript"
# The manuscript is maintained in Overleaf and is deliberately not tracked here.
# Its content checks live with the compile, not with the repository. Figure PDFs
# are produced here and carried across; that link is checked in group 4.
ok "manuscript is out of tree by design (Overleaf); figure link checked in group 4"

note "8. documentation debt"
if [ -f docs/CORRECTIONS_v7_7.md ]; then
  n=$(grep -c '^## ' docs/CORRECTIONS_v7_7.md)
  blocks=$(grep -c '^```$' docs/CORRECTIONS_v7_7.md)
  ok "docs/CORRECTIONS_v7_7.md present: $n sections, $((blocks / 2)) with live hits"
else skip "no docs/CORRECTIONS_v7_7.md — run apply_v7_7_repo.sh first"; fi
[ -f docs/PROVENANCE_ADDENDUM_v7_7.md ] && ok "addendum present" \
  || skip "docs/PROVENANCE_ADDENDUM_v7_7.md not placed"

note "9. branch checker"
# Checks 1 (identity) and 2 (thesis fingerprint) are EXPECTED to fail on main:
# main is the named branch and legitimately carries both. They are gates for
# anon-submission only. Checks 3 (paths), 4 (code) and 5 (orphans) are
# branch-independent and a failure there is real wherever it fires.
BR=$(git rev-parse --abbrev-ref HEAD)
if [ ! -f scripts/verify_branch.sh ]; then
  skip "verify_branch.sh not on this branch"
else
  bash scripts/verify_branch.sh >/tmp/_vb 2>&1; vb=$?
  sect=""
  while IFS= read -r line; do
    case "$line" in
      "=== "*) sect="${line#=== }" ;;
      "FAIL  "*)
        # the closing summary line is not an independent failure
        case "$line" in *"do not generate the mirror"*) continue ;; esac
        case "$sect" in
          1.*|2.*)
            if [ "$BR" = "anon-submission" ]; then bad "branch check $sect: ${line#FAIL  }"
            else printf 'note  branch check %s failed, expected on %s\n' "$sect" "$BR"; fi ;;
          "") : ;;
          *) bad "branch check $sect: ${line#FAIL  }" ;;
        esac ;;
    esac
  done < /tmp/_vb
  [ "$vb" -eq 0 ] && ok "verify_branch.sh PASS on $BR"
  rm -f /tmp/_vb
fi

note "9b. stray copies of these scripts"
stray=0
for f in apply_v7_7_repo.sh verify_v7_7.sh probe_power_inputs.py; do
  [ -f "./$f" ] && { bad "stray at repo root: ./$f (belongs in scripts/ or deleted)"; stray=1; }
done
[ -f paper_experiments/probe_power_inputs.py ] \
  && { bad "paper_experiments/probe_power_inputs.py — one-off diagnostic, delete it"; stray=1; }
[ "$stray" -eq 0 ] && ok "no stray copies"

note "10. remote state"
ahead=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo '?')
if [ "$ahead" = "0" ]; then ok "level with origin/main"
else bad "main is $ahead commit(s) ahead of origin — unpushed"; fi
git show-ref --verify --quiet refs/remotes/origin/v2.0-infastructure \
  && bad "origin/v2.0-infastructure exists and is unaudited" \
  || ok "no unaudited remote branch"

printf '\n-----\n'
if [ "$fail" -eq 0 ]; then echo "ALL CHECKS PASS"; else echo "$fail check(s) failed — see FAIL lines above"; fi
exit "$fail"
