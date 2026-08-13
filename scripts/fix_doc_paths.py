#!/usr/bin/env python3
"""Clear the remaining verify_branch.sh Check 3 failures.

    python scripts/fix_doc_paths.py            # dry run
    python scripts/fix_doc_paths.py --write

Three kinds of edit:

  RENAME     a backticked path that is simply wrong. `_v3` is the OUTPUT of
             scripts/make_fig1_gap_forest.py, not its name (ledger #39, #53b).
  PREFIX     a bare script name that lives under scripts/.
  UNQUOTE    setup instructions and prose, not repository paths. Backticks
             assert existence and Check 3 tests every backticked path
             (PROVENANCE.md Rule 8). These four are NOT defects; the previous
             advice to delete them would have removed working documentation.

Also drops the two Part 6 rows for notebooks now untracked.

Backups are written to <file>.prefix.bak and must be removed before commit --
ledger #49 records a tracked .bak as a defect.
"""
import argparse, difflib, sys
from pathlib import Path

EDITS = {
    "README.md": [
        ("RENAME",  "`make_fig1_gap_forest_v3.py`", "`scripts/make_fig1_gap_forest.py`"),
        ("RENAME",  "python make_fig1_gap_forest_v3.py", "python scripts/make_fig1_gap_forest.py"),
        ("UNQUOTE", "(`abstracts.tsv`, `neo.json`, `med-stopwords.txt`)",
                    "(abstracts.tsv, neo.json, med-stopwords.txt)"),
    ],
    "data/README.md": [
        ("RENAME",  "`scripts/make_fig1_gap_forest_v3.py`", "`scripts/make_fig1_gap_forest.py`"),
        ("RENAME",  "python scripts/make_fig1_gap_forest_v3.py", "python scripts/make_fig1_gap_forest.py"),
        ("RENAME",  "`make_fig1_gap_forest_v3.py`", "`scripts/make_fig1_gap_forest.py`"),
        ("PREFIX",  "`bootstrap_bert_per_fold.py`", "`scripts/bootstrap_bert_per_fold.py`"),
        ("UNQUOTE", "(`abstracts.tsv`, `neo.json`, `med-stopwords.txt`)",
                    "(abstracts.tsv, neo.json, med-stopwords.txt)"),
    ],
    "REPRODUCING.md": [
        # the real file exists; the ellipsis was an abbreviation, not a path
        ("RENAME",  "`..._smoke_rerun2.txt`", "`bow_statins_smoke_rerun2.txt`"),
    ],
    "PROVENANCE.md": [
        ("DROP", "| `archive/Main Classify Abstracts Code.ipynb` | Thesis-era notebook, full classification pipeline as submitted (2023) |\n", ""),
        ("DROP", "| `archive/Ontology Preferred Label Groupings.ipynb` | Thesis-era NEO ontology processing |\n", ""),
    ],
}

NOTE = """
STILL YOURS -- prose, not a path
  README.md:72 and data/README.md:72 describe the renamed script as rendering
  "the canonical Figure 1 forest plot". Ledger #53b: it writes
  fig1_gap_forest_v3.pdf, which nothing consumes. The shipped figure comes from
  scripts/make_paper_artifacts.py. Renaming the path clears Check 3 but leaves
  that claim wrong. Same for the "canonical regeneration paths" sentence at
  line 196 in both files.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if not Path("PROVENANCE.md").exists():
        sys.exit("run from the repository root")

    total = 0
    for fname, edits in EDITS.items():
        p = Path(fname)
        if not p.exists():
            print(f"SKIP  {fname} not found")
            continue
        orig = p.read_text(encoding="utf-8")
        s, applied = orig, []
        for kind, find, repl in edits:
            n = s.count(find)
            if n:
                s = s.replace(find, repl)
                applied.append(f"{kind:8} {n}x  {find[:64]}")
            else:
                applied.append(f"{'MISS':8} --   {find[:64]}")
        print(f"\n=== {fname}")
        for x in applied:
            print("  " + x)
        if s == orig:
            continue
        total += 1
        if a.write:
            p.with_suffix(p.suffix + ".prefix.bak").write_text(orig, encoding="utf-8")
            p.write_text(s, encoding="utf-8")
            print(f"  -> written (backup {p.name}.prefix.bak)")
        else:
            for line in difflib.unified_diff(orig.splitlines(), s.splitlines(),
                                             fname, fname + " (proposed)",
                                             lineterm="", n=0):
                if line.startswith(("+++", "---", "@@")):
                    continue
                print("     " + line)

    print(NOTE)
    if not a.write:
        print(f"dry run: {total} file(s) would change. Rerun with --write")
    else:
        print("next:")
        print("  rm -f *.prefix.bak data/*.prefix.bak      # ledger #49")
        print("  bash scripts/step1_docs.sh manifest")
        print("  bash scripts/verify_branch.sh")


if __name__ == "__main__":
    main()
