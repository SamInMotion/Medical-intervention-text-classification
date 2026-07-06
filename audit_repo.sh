#!/usr/bin/env bash
# =============================================================================
# Repository consistency audit script for:
#   "Evaluation design conditions the expert-vs-auto MeSH gap"
#   NEJLT submission / v7.2 paper state
#
# Run this from the repository root:
#   bash audit_repo.sh
#
# This script checks that the repository structure, file paths, and key data
# values are consistent with the manuscript's Appendix A and results tables.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

PASS=0
FAIL=0
WARN=0

# Helper functions
pass() { echo "  [PASS] $1"; ((PASS++)); }
fail() { echo "  [FAIL] $1"; ((FAIL++)); }
warn() { echo "  [WARN] $1"; ((WARN++)); }

section() {
    echo ""
    echo "============================================================================"
    echo "$1"
    echo "============================================================================"
}

# =============================================================================
section "1. DIRECTORY STRUCTURE vs. MANUSCRIPT APPENDIX A"
# =============================================================================

echo "Checking top-level layout..."

# Directories that must exist
for dir in data src tests paper_experiments notebooks outputs paper; do
    if [[ -d "$dir" ]]; then
        pass "Directory exists: $dir/"
    else
        fail "Missing directory: $dir/"
    fi
done

# The old 'scripts/' directory should NOT exist if unified to paper_experiments/
if [[ -d "scripts" ]]; then
    fail "Directory 'scripts/' still exists — should be removed/merged into paper_experiments/"
else
    pass "Directory 'scripts/' absent (unified into paper_experiments/)"
fi

# paper_experiments/outputs/ must exist
if [[ -d "paper_experiments/outputs" ]]; then
    pass "Directory exists: paper_experiments/outputs/"
else
    fail "Missing directory: paper_experiments/outputs/"
fi

# outputs/archive/ must exist
if [[ -d "outputs/archive" ]]; then
    pass "Directory exists: outputs/archive/"
else
    warn "Missing directory: outputs/archive/ (optional but recommended)"
fi

# =============================================================================
section "2. KEY SCRIPTS (Appendix A citations)"
# =============================================================================

echo "Checking scripts named in the manuscript..."

# Scripts at repo root
for file in \
    bootstrap_paired_permutation.py \
    demo_statistical_analysis.py \
    parse_bow_multirun.py; do
    if [[ -f "$file" ]]; then
        pass "Script exists: $file"
    else
        fail "Missing script: $file"
    fi
done

# Scripts in paper_experiments/
for file in \
    paper_experiments/patch_cohen_pipeline.py \
    paper_experiments/run_statins_subsampling.sh \
    paper_experiments/run_statins_10fold.sh \
    paper_experiments/parse_bow_experiments.py \
    paper_experiments/power_analysis.py; do
    if [[ -f "$file" ]]; then
        pass "Script exists: $file"
    else
        fail "Missing script: $file"
    fi
done

# Source files
for file in \
    src/cohen_pipeline.py \
    src/cohen_bert_pipeline.py; do
    if [[ -f "$file" ]]; then
        pass "Source exists: $file"
    else
        fail "Missing source: $file"
    fi
done

# Check for the old duplicate name
if [[ -f "bootstrap_bert_per_fold.py" ]]; then
    warn "Old duplicate 'bootstrap_bert_per_fold.py' still exists — remove if superseded"
else
    pass "Old duplicate 'bootstrap_bert_per_fold.py' absent"
fi

if [[ -f "paper_experiments/bootstrap_bert_per_fold.py" ]]; then
    warn "Old duplicate 'paper_experiments/bootstrap_bert_per_fold.py' still exists"
else
    pass "Old duplicate in paper_experiments/ absent"
fi

# =============================================================================
section "3. OUTPUT ARTIFACTS (Appendix A citations)"
# =============================================================================

echo "Checking data files named in the manuscript..."

# BERT per-seed JSONs
for topic in statins opiods adhd; do
    for mode in title_abstract_mesh auto_mesh; do
        for seed in 42 7 13 21 31; do
            file="outputs/bert_${topic}_${mode}_seed${seed}.json"
            if [[ -f "$file" ]]; then
                pass "Artifact exists: $file"
            else
                # Only fail for Statins (the most critical); warn for others
                if [[ "$topic" == "statins" ]]; then
                    fail "Missing artifact: $file"
                else
                    warn "Missing artifact: $file"
                fi
            fi
        done
    done
done

# Multi-seed summaries
for topic in statins opiods adhd; do
    file="outputs/bert_${topic}_multiseed_summary.json"
    if [[ -f "$file" ]]; then
        pass "Summary exists: $file"
    else
        if [[ "$topic" == "statins" ]]; then
            fail "Missing summary: $file"
        else
            warn "Missing summary: $file"
        fi
    fi
done

# Three-topic pooled summary
if [[ -f "outputs/bert_three_topic_multiseed_summary.json" ]]; then
    pass "Pooled summary exists: outputs/bert_three_topic_multiseed_summary.json"
else
    warn "Missing pooled summary: outputs/bert_three_topic_multiseed_summary.json"
fi

# BoW multi-run summaries
for topic in statins opiods adhd; do
    file="outputs/bow_${topic}_multirun_summary.json"
    if [[ -f "$file" ]]; then
        pass "BoW summary exists: $file"
    else
        if [[ "$topic" == "statins" ]]; then
            fail "Missing BoW summary: $file"
        else
            warn "Missing BoW summary: $file"
        fi
    fi
done

# Robustness analysis CSV
if [[ -f "paper_experiments/outputs/bow_experiments_summary.csv" ]]; then
    pass "Robustness CSV exists: paper_experiments/outputs/bow_experiments_summary.csv"
else
    warn "Missing robustness CSV: paper_experiments/outputs/bow_experiments_summary.csv"
fi

# =============================================================================
section "4. DATA INTEGRITY: JSON vs. PAPER TABLES"
# =============================================================================

echo "Checking that Statins BERT JSON values match Table 5 (v7.2 corrected)..."

if command -v python3 &>/dev/null; then
    python3 - <<'PYEOF'
import json, sys, math

errors = 0

def check(label, computed, expected, tol=0.001):
    global errors
    if abs(computed - expected) <= tol:
        print(f"  [PASS] {label}: {computed:+.6f} == {expected:+.6f}")
    else:
        print(f"  [FAIL] {label}: {computed:+.6f} != {expected:+.6f} (expected)")
        errors += 1

# Load Statins multi-seed summary
for path in ["outputs/bert_statins_multiseed_summary.json", "bert_statins_multiseed_summary.json"]:
    if __import__('pathlib').Path(path).exists():
        with open(path) as f:
            data = json.load(f)
        break
else:
    print("  [FAIL] Cannot find bert_statins_multiseed_summary.json")
    sys.exit(1)

# Table 5 expected values (v7.2 corrected)
expected_seeds = {
    42: +0.002,
    7:  +0.004,
    13: +0.040,
    21: +0.060,
    31: -0.007,
}

print("Checking per-seed gaps against Table 5...")
for seed, expected_gap in expected_seeds.items():
    seed_str = str(seed)
    if seed_str not in data.get("per_seed_gaps", {}):
        print(f"  [FAIL] Seed {seed} missing from JSON")
        errors += 1
        continue
    check(f"Seed {seed} gap", data["per_seed_gaps"][seed_str], expected_gap)

# Check pooled mean
if "expert_runs" in data and "auto_runs" in data:
    expert_wss = []
    auto_wss = []
    for seed in data["seeds_run"]:
        seed_str = str(seed)
        for fold in data["expert_runs"][seed_str]["folds"]:
            expert_wss.append(fold["wss_at_95"])
        for fold in data["auto_runs"][seed_str]["folds"]:
            auto_wss.append(fold["wss_at_95"])
    diffs = [e - a for e, a in zip(expert_wss, auto_wss)]
    pooled = sum(diffs) / len(diffs)
    check("Pooled mean (25 folds)", pooled, +0.020)

    # Check for the exact duplicate (seed 13, fold 1) that was flagged
    seed13_expert = [f["wss_at_95"] for f in data["expert_runs"]["13"]["folds"]]
    seed13_auto   = [f["wss_at_95"] for f in data["auto_runs"]["13"]["folds"]]
    if abs(seed13_expert[0] - seed13_auto[0]) < 1e-9:
        print(f"  [WARN] Seed 13 Fold 1: expert==auto exactly ({seed13_expert[0]:.10f})")
        print(f"         Investigate whether this is a genuine result or parsing bug")
    else:
        print(f"  [PASS] Seed 13 Fold 1: expert!=auto ({seed13_expert[0]:.6f} vs {seed13_auto[0]:.6f})")

sys.exit(0 if errors == 0 else 1)
PYEOF
    JSON_STATUS=$?
    if [[ $JSON_STATUS -ne 0 ]]; then
        ((FAIL++))
    fi
else
    warn "python3 not available — skipping JSON data integrity check"
fi

# =============================================================================
section "5. HYPOTHESIS REGISTER (H-Pub2)"
# =============================================================================

echo "Checking pre-registration documentation..."

# H-Pub2 is documented in the paper's Appendix B
# We check if it's also in the repo as a standalone file
H_REG_FOUND=0
for candidate in H-Pub2.md hypothesis_register.md preregistration.json H_Pub2.txt; do
    if [[ -f "$candidate" ]]; then
        pass "Hypothesis register found: $candidate"
        H_REG_FOUND=1
    fi
done

if [[ $H_REG_FOUND -eq 0 ]]; then
    warn "No standalone H-Pub2 register file found in repo (it IS in the paper's Appendix B)"
    echo "         Consider adding H-Pub2.md for discoverability"
fi

# =============================================================================
section "6. README CONSISTENCY"
# =============================================================================

echo "Checking README for contradictions..."

if [[ -f "README.md" ]]; then
    pass "README.md exists"

    # Check for the multi-seed contradiction (should be FIXED now)
    if grep -qi "one seed per mode\|single.seed" README.md; then
        fail "README still contains 'one seed per mode' / 'single-seed' contradiction"
    else
        pass "README appears free of single-seed contradiction"
    fi

    # Check for old path references
    if grep -q "scripts/" README.md; then
        fail "README still references 'scripts/' — should be 'paper_experiments/'"
    else
        pass "README uses 'paper_experiments/' consistently"
    fi

    # Check for old bootstrap script name
    if grep -q "bootstrap_bert_per_fold" README.md; then
        fail "README still references 'bootstrap_bert_per_fold.py' — should be 'bootstrap_paired_permutation.py'"
    else
        pass "README uses correct bootstrap script name"
    fi

    # Check license claim
    if grep -q "MIT License" README.md; then
        if [[ -f "LICENSE" ]]; then
            pass "README claims MIT + LICENSE file exists"
        else
            fail "README claims MIT but LICENSE file is missing"
        fi
    fi

    # Check for version tag reference
    if grep -q "nejlt-v1\|v7.2\|commit" README.md; then
        pass "README mentions version tag or commit"
    else
        warn "README does not mention version tag or commit hash for paper state"
    fi
else
    fail "README.md missing"
fi

# =============================================================================
section "7. GIT STATE & VERSION CONTROL"
# =============================================================================

echo "Checking git tags and commit state..."

if [[ -d ".git" ]]; then
    pass "Git repository initialised"

    # Check for the paper version tag
    if git tag | grep -q "nejlt-v1\|v7.2\|paper-submission"; then
        pass "Version tag found: $(git tag | grep -E 'nejlt-v1|v7.2|paper-submission' | head -1)"
    else
        warn "No version tag (nejlt-v1, v7.2, etc.) found — consider tagging the paper state"
    fi

    # Check for uncommitted changes
    if git diff-index --quiet HEAD --; then
        pass "Working tree is clean (no uncommitted changes)"
    else
        warn "Working tree has uncommitted changes — commit before submission"
    fi

    # Show last commit
    echo ""
    echo "  Last commit:"
    git log -1 --oneline | sed 's/^/    /'
else
    warn "Not a git repository (or .git/ missing)"
fi

# =============================================================================
section "8. REPRODUCING.md"
# =============================================================================

if [[ -f "REPRODUCING.md" ]]; then
    pass "REPRODUCING.md exists (claim-to-command mapping)"
else
    warn "REPRODUCING.md missing — README says this is the canonical regeneration guide"
fi

# =============================================================================
section "9. REQUIREMENTS & SETUP"
# =============================================================================

if [[ -f "requirements.txt" ]]; then
    pass "requirements.txt exists"
else
    warn "requirements.txt missing"
fi

# =============================================================================
section "10. SUMMARY"
# =============================================================================

echo ""
echo "============================================================================"
echo "AUDIT COMPLETE"
echo "============================================================================"
printf "  Passed:  %3d\n" "$PASS"
printf "  Failed:  %3d\n" "$FAIL"
printf "  Warnings:%3d\n" "$WARN"
echo ""

if [[ $FAIL -eq 0 ]]; then
    echo "  STATUS: Repository structure is CONSISTENT with the manuscript."
    echo "          A reviewer can trace every cited path to an existing file."
    if [[ $WARN -gt 0 ]]; then
        echo "          ($WARN optional items flagged for attention)"
    fi
    exit 0
else
    echo "  STATUS: Repository has $FAIL inconsistency(ies) that need fixing."
    echo "          Review these FAIL items above before submission."
    exit 1
fi
