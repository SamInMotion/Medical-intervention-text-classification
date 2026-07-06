#!/usr/bin/env bash
# ============================================================================
# state_diagnostic.sh
# ----------------------------------------------------------------------------
# Read-only audit of repo state across three locations:
#   1. Local working tree (default: ~/projects/Medical-intervention-text-classification)
#   2. Google Drive folder used by the Colab notebook (default: /g/My Drive/cohen_bert_run)
#   3. GitHub remote (SamInMotion/Medical-intervention-text-classification)
#
# Usage:
#   bash state_diagnostic.sh                 # uses defaults
#   bash state_diagnostic.sh /c/path/repo "/g/My Drive/cohen_bert_run"
#
# Output: state_diagnostic_YYYYMMDD_HHMMSS.txt in the current directory.
# Safe: no writes, no git modifications, no network beyond GitHub raw reads.
# ============================================================================

set -u

LOCAL_REPO="${1:-$HOME/projects/Medical-intervention-text-classification}"
DRIVE_DIR="${2:-/g/My Drive/cohen_bert_run}"
GH_OWNER="SamInMotion"
GH_REPO="Medical-intervention-text-classification"
GH_BRANCH_GUESS="${3:-main}"  # script will auto-detect master vs main

# Files we care about
KEY_FILES_SRC=(
  "src/features.py"
  "src/preprocessing.py"
  "src/cohen_pipeline.py"
  "src/cohen_bert_pipeline.py"
  "src/benchmark_loader.py"
  "src/bert_models.py"
  "src/evaluation.py"
  "src/models.py"
  "src/config.py"
  "src/auto_mesh.py"
)

OUT="state_diagnostic_$(date +%Y%m%d_%H%M%S).txt"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
section() {
  printf '\n\n========================================================================\n'  >> "$OUT"
  printf '== %s\n'                                                                       "$1" >> "$OUT"
  printf '========================================================================\n'    >> "$OUT"
}
subsection() {
  printf '\n--- %s ---\n' "$1" >> "$OUT"
}
note() {
  printf '%s\n' "$1" >> "$OUT"
}
run() {
  # Run a command, capture stdout+stderr to the report, never fail the script
  printf '\n$ %s\n' "$*" >> "$OUT"
  "$@" >> "$OUT" 2>&1 || printf '[command exited with status %s — continuing]\n' "$?" >> "$OUT"
}
hash_file() {
  # SHA-256 of a file, plus byte size and line count. Works with available tools.
  local p="$1"
  if [ ! -f "$p" ]; then
    printf '   [missing]\n' >> "$OUT"
    return
  fi
  local sz lines hash
  sz=$(wc -c < "$p" 2>/dev/null | tr -d '[:space:]')
  lines=$(wc -l < "$p" 2>/dev/null | tr -d '[:space:]')
  if command -v sha256sum >/dev/null 2>&1; then
    hash=$(sha256sum "$p" | awk '{print $1}')
  elif command -v shasum >/dev/null 2>&1; then
    hash=$(shasum -a 256 "$p" | awk '{print $1}')
  elif command -v openssl >/dev/null 2>&1; then
    hash=$(openssl dgst -sha256 "$p" | awk '{print $NF}')
  else
    hash="(no sha256 tool available)"
  fi
  printf '   sha256=%s  size=%sB  lines=%s\n' "$hash" "$sz" "$lines" >> "$OUT"
}
dump_small_file() {
  # Dump the full contents of a small file. Used for features.py and preprocessing.py.
  local p="$1"
  if [ ! -f "$p" ]; then
    printf '   [missing — cannot dump]\n' >> "$OUT"
    return
  fi
  printf '\n>>>>> BEGIN FILE: %s <<<<<\n' "$p" >> "$OUT"
  cat "$p" >> "$OUT"
  printf '\n<<<<< END FILE: %s >>>>>\n' "$p" >> "$OUT"
}
grep_marker() {
  # Search a file for a marker string, print yes/no + first matching line(s).
  local p="$1" pattern="$2"
  if [ ! -f "$p" ]; then
    printf '   pattern "%s": [file missing]\n' "$pattern" >> "$OUT"
    return
  fi
  if grep -q "$pattern" "$p" 2>/dev/null; then
    local hit
    hit=$(grep -n "$pattern" "$p" 2>/dev/null | head -3)
    printf '   pattern "%s": PRESENT\n%s\n' "$pattern" "$hit" >> "$OUT"
  else
    printf '   pattern "%s": absent\n' "$pattern" >> "$OUT"
  fi
}

# ----------------------------------------------------------------------------
# Start
# ----------------------------------------------------------------------------
: > "$OUT"
section "STATE DIAGNOSTIC — $(date)"
note   "Local repo path:  $LOCAL_REPO"
note   "Drive directory:  $DRIVE_DIR"
note   "GitHub remote:    https://github.com/$GH_OWNER/$GH_REPO"
note   ""
note   "Run via: bash state_diagnostic.sh [LOCAL] [DRIVE] [BRANCH]"
note   "All commands are read-only."

# ----------------------------------------------------------------------------
# 1. Environment
# ----------------------------------------------------------------------------
section "1. ENVIRONMENT"
run uname -a
run bash --version
run python --version
run python -c "import sys; print(sys.executable)"
run pwd

# Check tools we'll use
subsection "Tool availability"
for tool in git curl sha256sum shasum openssl wc grep; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf '   %-12s %s\n' "$tool" "$(command -v "$tool")" >> "$OUT"
  else
    printf '   %-12s NOT FOUND\n' "$tool" >> "$OUT"
  fi
done

# ----------------------------------------------------------------------------
# 2. Local repo discovery
# ----------------------------------------------------------------------------
section "2. LOCAL REPO"

if [ ! -d "$LOCAL_REPO" ]; then
  note "Local repo path does not exist: $LOCAL_REPO"
  note "Override by passing the actual path as the first argument."
  note "Scanning common parent directories for the repo name..."
  for parent in "$HOME" "$HOME/projects" "$HOME/repos" "$HOME/code" "$HOME/Documents" "$HOME/Desktop" "/c" "/d"; do
    if [ -d "$parent" ]; then
      found=$(find "$parent" -maxdepth 4 -type d -name "Medical-intervention-text-classification" 2>/dev/null | head -5)
      if [ -n "$found" ]; then
        note "Found candidate paths under $parent:"
        printf '%s\n' "$found" >> "$OUT"
      fi
    fi
  done
else
  cd "$LOCAL_REPO" || exit 0
  note "Working dir: $(pwd)"

  subsection "Git status / branch / remote"
  run git rev-parse --abbrev-ref HEAD
  run git remote -v
  run git status --short
  run git log -10 --oneline --decorate --all

  subsection "All branches (local and remote-tracking)"
  run git branch -a

  subsection "Diff against origin (uncommitted + unpushed)"
  run git fetch --all --tags --quiet
  run git status -b --porcelain=v1
  run git log --oneline @{u}..HEAD 2>/dev/null
  run git log --oneline HEAD..@{u} 2>/dev/null

  subsection "Search for ALL files named features.py or preprocessing.py anywhere"
  run find . -type f \( -name "features.py" -o -name "preprocessing.py" \) -not -path "./.git/*"

  subsection "Git history for src/features.py (last 20 commits)"
  run git log -20 --oneline --follow -- src/features.py

  subsection "Git history for src/preprocessing.py (last 20 commits)"
  run git log -20 --oneline --follow -- src/preprocessing.py

  subsection "Key file inventory (hashes)"
  for f in "${KEY_FILES_SRC[@]}"; do
    printf '\n%s\n' "$f" >> "$OUT"
    hash_file "$f"
  done

  subsection "Search for CountVectorizer usage anywhere in the working tree"
  run grep -rn "CountVectorizer" --include="*.py" .

  subsection "Search for Keras Tokenizer usage"
  run grep -rn "keras.*Tokenizer\|keras_text\|preprocessing.text" --include="*.py" .

  subsection "Search for set_seeds / NUMPY_SEED / SPLIT_SEED in config and scripts"
  run grep -rn "NUMPY_SEED\|TF_SEED\|SPLIT_SEED\|set_seeds" --include="*.py" .

  subsection "FULL DUMP — src/features.py (LOCAL)"
  dump_small_file "src/features.py"

  subsection "FULL DUMP — src/preprocessing.py (LOCAL)"
  dump_small_file "src/preprocessing.py"

  subsection "Markers in src/features.py"
  grep_marker "src/features.py" "CountVectorizer"
  grep_marker "src/features.py" "ngram"
  grep_marker "src/features.py" "max_features"
  grep_marker "src/features.py" "num_words"
  grep_marker "src/features.py" "keras"

  subsection "Markers in src/cohen_pipeline.py"
  grep_marker "src/cohen_pipeline.py" "subsample"
  grep_marker "src/cohen_pipeline.py" "load_cohen_topic"
  grep_marker "src/cohen_pipeline.py" "StratifiedKFold"

  subsection "Scripts directory (multi-run BoW)"
  if [ -d "scripts" ]; then
    run ls -la scripts/
  else
    note "No scripts/ directory in repo root."
  fi

  subsection "Recent files modified in the last 14 days (src/, scripts/, top-level)"
  run find . -maxdepth 3 -type f \( -name "*.py" -o -name "*.sh" -o -name "*.md" -o -name "*.txt" \) -mtime -14 -not -path "./.git/*" -not -path "./data/*" -not -path "./.venv*/*"

  subsection "requirements.txt or pyproject.toml"
  if [ -f "requirements.txt" ]; then
    run cat requirements.txt
  fi
  if [ -f "pyproject.toml" ]; then
    run cat pyproject.toml
  fi

  subsection "Virtual env presence (.venv* directories)"
  run ls -d .venv* 2>/dev/null
fi

# ----------------------------------------------------------------------------
# 3. Google Drive
# ----------------------------------------------------------------------------
section "3. GOOGLE DRIVE — cohen_bert_run folder"

if [ ! -d "$DRIVE_DIR" ]; then
  note "Drive directory not found at: $DRIVE_DIR"
  note "If Drive is mounted at a different letter, pass it as second arg:"
  note "   bash state_diagnostic.sh \"\" \"/h/My Drive/cohen_bert_run\""
else
  subsection "Top-level listing"
  run ls -la "$DRIVE_DIR"

  subsection "All .py files in Drive folder (recursive, depth 3)"
  run find "$DRIVE_DIR" -maxdepth 3 -type f -name "*.py"

  subsection "All .ipynb files"
  run find "$DRIVE_DIR" -maxdepth 3 -type f -name "*.ipynb"

  subsection "Hashes of any Python files at Drive root"
  while IFS= read -r f; do
    printf '\n%s\n' "$f" >> "$OUT"
    hash_file "$f"
  done < <(find "$DRIVE_DIR" -maxdepth 1 -type f -name "*.py")

  for candidate in "features.py" "preprocessing.py" "cohen_pipeline.py" "cohen_bert_pipeline.py" "bert_models.py"; do
    found_in_drive=$(find "$DRIVE_DIR" -maxdepth 3 -type f -name "$candidate" 2>/dev/null | head -3)
    if [ -n "$found_in_drive" ]; then
      while IFS= read -r p; do
        subsection "FULL DUMP — Drive copy: $p"
        if [ "$candidate" = "features.py" ] || [ "$candidate" = "preprocessing.py" ]; then
          dump_small_file "$p"
        else
          run head -80 "$p"
          note "[... truncated to first 80 lines for brevity ...]"
          hash_file "$p"
        fi
      done <<< "$found_in_drive"
    fi
  done
fi

# ----------------------------------------------------------------------------
# 4. GitHub remote (raw content)
# ----------------------------------------------------------------------------
section "4. GITHUB REMOTE — raw content from default branch"

if ! command -v curl >/dev/null 2>&1; then
  note "curl not available — skipping GitHub fetch."
else
  # Detect default branch
  default_branch=""
  api_url="https://api.github.com/repos/$GH_OWNER/$GH_REPO"
  api_resp=$(curl -fsSL "$api_url" 2>/dev/null)
  if [ -n "$api_resp" ]; then
    default_branch=$(printf '%s' "$api_resp" | grep -o '"default_branch": *"[^"]*"' | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
  fi
  if [ -z "$default_branch" ]; then
    default_branch="$GH_BRANCH_GUESS"
    note "Could not auto-detect default branch via API. Falling back to: $default_branch"
  else
    note "GitHub default branch: $default_branch"
  fi

  subsection "Repository metadata"
  printf '%s\n' "$api_resp" | head -40 >> "$OUT"

  subsection "Latest commit on default branch"
  curl -fsSL "https://api.github.com/repos/$GH_OWNER/$GH_REPO/commits/$default_branch" 2>/dev/null | head -50 >> "$OUT"

  subsection "Branches on remote"
  curl -fsSL "https://api.github.com/repos/$GH_OWNER/$GH_REPO/branches" 2>/dev/null \
    | grep -E '"name"|"sha"' | head -40 >> "$OUT"

  subsection "Tree contents — src/"
  curl -fsSL "https://api.github.com/repos/$GH_OWNER/$GH_REPO/contents/src?ref=$default_branch" 2>/dev/null \
    | grep -E '"name"|"size"|"sha"' >> "$OUT"

  subsection "Hashes + dumps of remote key files"
  TMPDIR=$(mktemp -d 2>/dev/null || echo "/tmp/state_diag_$$")
  mkdir -p "$TMPDIR"
  for f in "${KEY_FILES_SRC[@]}"; do
    url="https://raw.githubusercontent.com/$GH_OWNER/$GH_REPO/$default_branch/$f"
    out_path="$TMPDIR/$(basename "$f")"
    if curl -fsSL "$url" -o "$out_path" 2>/dev/null && [ -s "$out_path" ]; then
      printf '\n%s (REMOTE %s)\n' "$f" "$default_branch" >> "$OUT"
      hash_file "$out_path"
      if [ "$(basename "$f")" = "features.py" ] || [ "$(basename "$f")" = "preprocessing.py" ]; then
        subsection "FULL DUMP — GitHub copy: $f"
        dump_small_file "$out_path"
      fi
    else
      printf '\n%s — NOT FOUND on remote at branch %s\n' "$f" "$default_branch" >> "$OUT"
    fi
  done

  # Also check the other obvious branch names in case features.py was patched on a side branch
  subsection "Check side branches for features.py (cohen-bert, benchmark, patch, paper, multi-run)"
  for br in cohen-bert benchmark patch paper multi-run multirun cohen-benchmark statins-multirun bow-patch; do
    url="https://raw.githubusercontent.com/$GH_OWNER/$GH_REPO/$br/src/features.py"
    if curl -fsI "$url" 2>/dev/null | head -1 | grep -q "200"; then
      printf '\nBranch %s — features.py exists\n' "$br" >> "$OUT"
      tmp="$TMPDIR/features_${br}.py"
      curl -fsSL "$url" -o "$tmp"
      hash_file "$tmp"
      if grep -q "CountVectorizer" "$tmp" 2>/dev/null; then
        printf '   CONTAINS CountVectorizer — this may be the patched version\n' >> "$OUT"
      fi
    fi
  done

  rm -rf "$TMPDIR" 2>/dev/null
fi

# ----------------------------------------------------------------------------
# 5. Cross-location hash comparison summary
# ----------------------------------------------------------------------------
section "5. SUMMARY"

note ""
note "Three things to look for when reading this report:"
note ""
note "  (a) Does any features.py anywhere contain 'CountVectorizer'?"
note "      Search this file for 'PRESENT' under markers for src/features.py."
note "      If absent everywhere, the 'patched features.py' described in"
note "      Consolidation v1/v2/v3 was never deployed and the BoW pipeline"
note "      runs on the Keras Tokenizer version as uploaded."
note ""
note "  (b) Do the local, Drive, and GitHub copies of features.py share the"
note "      same sha256? If yes, all three are in sync. If different, identify"
note "      which copy was used by the actual BoW multi-run on June 18–20."
note ""
note "  (c) Are there uncommitted local changes to src/features.py or"
note "      src/preprocessing.py? Check Section 2 git status."
note ""
note "Paste the entire output of this report back to me."

printf '\n\nReport saved to: %s\n' "$OUT"
printf 'Lines: %s   Size: %s bytes\n' "$(wc -l < "$OUT")" "$(wc -c < "$OUT")"
