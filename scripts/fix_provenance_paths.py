#!/usr/bin/env python3
"""Apply the six deterministic PROVENANCE.md fixes for branch checks 3 and 5.

    python scripts/fix_provenance_paths.py            # dry run, shows the diff
    python scripts/fix_provenance_paths.py --write    # apply

Why these six. verify_branch.sh Check 3 extracts every backticked path and tests
it. It cannot distinguish "this file is here" from "this path is the defect".
Two of the six are genuine typos; two are defect records that must lose their
backticks; one row cites a file that no longer exists separately; and Check 5
needs the six archive/ files given a Part 6 home.

Writes a .bak before modifying.
"""
import argparse, difflib, shutil, sys
from pathlib import Path

P = Path("PROVENANCE.md")

# (description, find, replace, expected_min_count)
EDITS = [
    ("Check 3: bare script name -> scripts/ path",
     "`apply_v7_7_repo.sh`", "`scripts/apply_v7_7_repo.sh`", 1),

    ("Check 3: bare script name -> scripts/ path",
     "| 40 | Appendix A.1 lists analysis scripts at paths that do not match the current tree | Partly resolved: `bootstrap_bert_per_fold.py` restored, see #54 |",
     "| 40 | Appendix A.1 lists analysis scripts at paths that do not match the current tree | Partly resolved: `scripts/bootstrap_bert_per_fold.py` restored, see #54 |",
     1),

    ("Check 3: defect record, drop backticks (#39)",
     "reference `make_fig1_gap_forest_v3.py` at root",
     "reference make_fig1_gap_forest_v3.py at root", 1),

    ("Check 3: defect record, drop backticks (#53b)",
     "`scripts/make_fig1_gap_forest.py` → `fig1_gap_forest_v3.pdf`",
     "`scripts/make_fig1_gap_forest.py` writes fig1_gap_forest_v3.pdf", 1),

    ("Check 3: defect record, drop backticks (#42)",
     "| 42 | `docs/Context_Update_188.md` and consolidation drafts",
     "| 42 | docs/Context_Update_188.md and consolidation drafts", 1),

    ("Check 3: addendum was consolidated into this file, drop its row",
     "| `docs/PROVENANCE_ADDENDUM_v7_7.md` | Source document for this merge; sections C, D, E remain live worklists |\n",
     "", 1),
]

ARCHIVE_ROWS = """| `archive/Main Classify Abstracts Code.ipynb` | Thesis-era notebook, full classification pipeline as submitted (2023) |
| `archive/Ontology Preferred Label Groupings.ipynb` | Thesis-era NEO ontology processing |
| `archive/fig1_gap_forest_v2.pdf` | Output of the decoy generator, see #53 |
| `archive/fig1_gap_forest_v2.png` | Output of the decoy generator, see #53 |
| `archive/commit_and_store.sh` | Superseded session helper |
| `archive/verify_setup.sh` | Superseded setup check. `paper_experiments/README_paper_experiments.md` still cites the pre-archive path; update it or restore the file |
"""

ANCHOR = "| `archive/pre_v7_7/` |"

RULE8 = """8. **A path recorded as a defect is written without backticks.** Added
   2026-08-11. Backticks assert the path exists, and `verify_branch.sh` Check 3
   tests every backticked path in this file.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    if not P.exists():
        sys.exit(f"{P} not found -- run from the repository root")
    orig = P.read_text(encoding="utf-8")
    s = orig
    applied, missed = [], []

    for desc, find, repl, n in EDITS:
        c = s.count(find)
        if c >= n:
            s = s.replace(find, repl)
            applied.append(f"{desc}  ({c}x)")
        else:
            missed.append(f"{desc}\n      looked for: {find[:88]}")

    # Check 5: archive rows into "Superseded, retained as record"
    if "archive/Main Classify Abstracts Code.ipynb`" in s:
        applied.append("Check 5: archive rows already present, skipped")
    elif ANCHOR in s:
        i = s.index(ANCHOR)
        j = s.index("\n", i) + 1
        s = s[:j] + ARCHIVE_ROWS + s[j:]
        applied.append("Check 5: six archive/ rows added to Part 6")
    else:
        missed.append(f"Check 5: anchor not found: {ANCHOR}")

    # Rule 8
    if "A path recorded as a defect is written without backticks" in s:
        applied.append("Rule 8 already present, skipped")
    elif "\n7. **Non-numeric claims get rows too.**" in s:
        k = s.index("\n7. **Non-numeric claims get rows too.**")
        end = s.index("\n\n", k) + 1
        s = s[:end] + RULE8 + s[end:]
        applied.append("Rule 8 added to Rules of use")
    else:
        missed.append("Rule 8: anchor not found (Rule 7 line)")

    print("APPLIED")
    for x in applied:
        print("  ok   " + x)
    if missed:
        print("\nNOT APPLIED -- edit by hand")
        for x in missed:
            print("  MISS " + x)

    if s == orig:
        print("\nno change")
        return

    if not a.write:
        print("\n--- diff (dry run; rerun with --write) ---")
        for line in difflib.unified_diff(
                orig.splitlines(), s.splitlines(),
                "PROVENANCE.md", "PROVENANCE.md (proposed)", lineterm="", n=1):
            print(line)
        return

    shutil.copy(P, P.with_suffix(".md.bak"))
    P.write_text(s, encoding="utf-8")
    print(f"\nwrote {P} (backup at {P.with_suffix('.md.bak')})")
    print("next: bash scripts/step1_docs.sh manifest && bash scripts/verify_branch.sh")


if __name__ == "__main__":
    main()
