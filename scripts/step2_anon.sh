#!/usr/bin/env bash
# step2_anon.sh — rebuild anon-submission from main.
#
#     bash scripts/step2_anon.sh delta      # what the current branch changes vs main
#     bash scripts/step2_anon.sh preflight  # gates that must pass before rebuilding
#     bash scripts/step2_anon.sh rebuild    # create the new branch (does not push)
#     bash scripts/step2_anon.sh verify     # run the five checks on the new branch
#
# The scrub is NOT automated. `delta` prints exactly what the existing branch
# removes and rewrites so the same edits can be reapplied deliberately. A script
# that guessed at the scrub would be the same class of error as a regex written
# against strings nobody had read.
#
# The old branch is kept as anon-submission-prev until you delete it.

set -u
cd "$(git rev-parse --show-toplevel)" 2>/dev/null || { echo "not a git repo" >&2; exit 2; }
CMD="${1:-preflight}"
OLD=anon-submission
BAK="anon-submission-prev-$(date +%Y%m%d)"

hdr()  { printf '\n\033[1m=== %s\033[0m\n' "$1"; }
bad()  { printf 'FAIL  %s\n' "$1"; }
ok()   { printf 'ok    %s\n' "$1"; }

case "$CMD" in

delta)
  hdr "files the branch DELETES relative to main"
  git diff --diff-filter=D --name-only main.."$OLD" | sed 's/^/  - /' || true
  hdr "files the branch ADDS relative to main"
  git diff --diff-filter=A --name-only main.."$OLD" | sed 's/^/  + /' || true
  hdr "files the branch MODIFIES relative to main"
  git diff --diff-filter=M --name-only main.."$OLD" | sed 's/^/  ~ /' || true
  hdr "the scrub, line by line, for each modified file"
  echo "  Review each before reapplying:"
  for f in $(git diff --diff-filter=M --name-only main.."$OLD"); do
    printf '\n  --- %s\n' "$f"
    git diff main.."$OLD" -- "$f" | sed -n '1,60p' | sed 's/^/      /'
  done
  hdr "branch point"
  echo "  merge-base main $OLD: $(git merge-base main "$OLD" 2>/dev/null || echo unknown)"
  echo "  $OLD head:            $(git rev-parse --short "$OLD" 2>/dev/null || echo unknown)"
  ;;

preflight)
  fail=0
  hdr "1. main is committed and pushed"
  [ -z "$(git status --porcelain)" ] && ok "working tree clean" || { bad "uncommitted changes"; git status --short | sed 's/^/      /'; fail=1; }
  a=$(git rev-list --count origin/main..main 2>/dev/null || echo '?')
  [ "$a" = "0" ] && ok "main level with origin" || { bad "main is $a commit(s) ahead of origin — push first"; fail=1; }

  hdr "2. main passes the branch-independent checks"
  if [ -f scripts/verify_branch.sh ]; then
    bash scripts/verify_branch.sh >/tmp/_vb 2>&1
    sect=""
    while IFS= read -r line; do
      case "$line" in
        "=== "*) sect="${line#=== }" ;;
        "FAIL  "*)
          case "$line" in *"do not generate the mirror"*) continue ;; esac
          case "$sect" in
            1.*|2.*) printf 'note  %s expected to fail on main\n' "$sect" ;;
            "") : ;;
            *) bad "$sect: ${line#FAIL  }"; fail=1 ;;
          esac ;;
      esac
    done < /tmp/_vb
    rm -f /tmp/_vb
  else bad "scripts/verify_branch.sh not found"; fail=1; fi

  hdr "3. manuscript is not tracked here"
  n=$(git ls-files 'paper/*.tex' '*.tex' | wc -l | tr -d ' ')
  [ "$n" -eq 0 ] && ok "no .tex tracked (manuscript lives in Overleaf)" \
    || printf 'note  %s .tex file(s) tracked — Check 2 will test them for FINGERPRINT\n' "$n"

  hdr "4. figures and outputs current"
  bash scripts/verify_v7_7.sh >/tmp/_v7 2>&1
  grep -E '^FAIL' /tmp/_v7 | sed 's/^/      /' || ok "verify_v7_7.sh clean"
  rm -f /tmp/_v7

  printf '\n-----\n'
  [ "$fail" -eq 0 ] && echo "PREFLIGHT PASS — safe to run: bash $0 rebuild" \
                    || echo "PREFLIGHT FAIL — do not rebuild yet"
  exit "$fail"
  ;;

rebuild)
  hdr "0. re-running preflight"
  bash "$0" preflight >/dev/null 2>&1 || { echo "preflight failed; run it and read the output" >&2; exit 1; }
  ok "preflight passed"

  hdr "1. preserving the current branch"
  if git show-ref --verify --quiet "refs/heads/$OLD"; then
    git branch -m "$OLD" "$BAK" && ok "renamed $OLD -> $BAK"
  else
    ok "no local $OLD to preserve"
  fi

  hdr "2. new branch from main"
  git checkout -q main && git checkout -q -b "$OLD" && ok "created $OLD at $(git rev-parse --short HEAD)"

  hdr "3. reapply the scrub"
  cat <<'SCRUB'
  NOT AUTOMATED. Reapply, using `bash scripts/step2_anon.sh delta` as the record
  of what the previous branch did. Expect at least:

    a  Remove identity-bearing tracked files. Ledger #42: docs/Context_Update_*.md
       and consolidation drafts. Check `git ls-files docs/`.
    b  Rewrite README.md and data/README.md to describe only what is on the
       branch (CU 213 Part 3).
    c  Scrub /g/My Drive/ paths and third-party names from paper_experiments/
       (ledger #58 notes Check 1 does not catch either).
    d  Decide docs/PROVENANCE_ADDENDUM_v7_7.md and docs/CORRECTIONS_v7_7.md:
       both name internal document identifiers (CU numbers, session dates).
    e  Regenerate MANIFEST.md after the removals:
           bash scripts/step1_docs.sh manifest

  Then:  bash scripts/step2_anon.sh verify
SCRUB
  ;;

verify)
  b=$(git rev-parse --abbrev-ref HEAD)
  [ "$b" = "$OLD" ] || { echo "FAIL  on branch '$b', expected '$OLD'" >&2; exit 1; }
  hdr "five checks on $OLD — all must pass"
  bash scripts/verify_branch.sh; rc=$?
  hdr "supplementary sweep (patterns Check 1 does not cover, ledger #58)"
  for p in 'My Drive' 'C:\\Users' 'Christer' '10\.5281/zenodo' 'arxiv\.org' 'Context_Update' ' CU [0-9]'; do
    h=$(git grep -c -iI -E "$p" -- . 2>/dev/null | grep -v '^scripts/verify_branch.sh' | head -5)
    if [ -n "$h" ]; then printf '  HIT  %-18s\n' "$p"; printf '%s\n' "$h" | sed 's/^/         /'
    else printf '  ok   %-18s\n' "$p"; fi
  done
  printf '\n-----\n'
  [ "$rc" -eq 0 ] && echo "BRANCH READY — regenerate the mirror in this sitting" \
                  || echo "NOT READY — resolve the FAIL lines above"
  echo "Mirror: note whether 4open.science issues a NEW URL. Appendix A.1 hardcodes the current one."
  exit "$rc"
  ;;

*) echo "usage: bash $0 {delta|preflight|rebuild|verify}" >&2; exit 2 ;;
esac
