#!/usr/bin/env bash
# ============================================================================
# verify_setup.sh
# ----------------------------------------------------------------------------
# Two checks (~30 seconds):
#   1. Local features.py and preprocessing.py match GitHub main
#   2. The v2.0-infastructure branch does not hide a CountVectorizer features.py
#
# Run from inside the local repo:
#   cd /c/Users/samue.KITAB/Medical-intervention-text-classification
#   bash paper_experiments/verify_setup.sh
# ============================================================================

set -u

GH_BASE="https://raw.githubusercontent.com/SamInMotion/Medical-intervention-text-classification"

echo "============================================================"
echo "1. Local vs GitHub main — features.py and preprocessing.py"
echo "============================================================"

for f in src/features.py src/preprocessing.py; do
  if [ ! -f "$f" ]; then
    echo "[MISSING LOCAL] $f"
    continue
  fi
  local_hash=$(sha256sum "$f" | awk '{print $1}')
  remote_hash=$(curl -fsSL "$GH_BASE/main/$f" | sha256sum | awk '{print $1}')
  if [ "$local_hash" = "$remote_hash" ]; then
    echo "[MATCH] $f"
  else
    echo "[DIFFER] $f"
    echo "         local:  $local_hash"
    echo "         remote: $remote_hash"
    echo "         Action: run 'git diff origin/main -- $f' to see what changed."
  fi
done

echo ""
echo "============================================================"
echo "2. v2.0-infastructure branch — features.py inspection"
echo "============================================================"

tmp=$(mktemp)
url="$GH_BASE/v2.0-infastructure/src/features.py"
if curl -fsSL "$url" -o "$tmp" 2>/dev/null && [ -s "$tmp" ]; then
  echo "Branch v2.0-infastructure has src/features.py:"
  echo "  sha256=$(sha256sum "$tmp" | awk '{print $1}')"
  echo "  lines=$(wc -l < "$tmp")"
  if grep -q "CountVectorizer" "$tmp"; then
    echo "  [!] Contains CountVectorizer — investigate before patching"
  else
    echo "  [OK] No CountVectorizer. Same family as main."
  fi
else
  echo "Branch v2.0-infastructure has no src/features.py at that path."
fi
rm -f "$tmp"

echo ""
echo "Done. If both sections show MATCH and no CountVectorizer warning,"
echo "proceed to: python paper_experiments/patch_cohen_pipeline.py"
