#!/usr/bin/env python
"""Generate MANIFEST.md: why every tracked file is in the repository.

Nothing here is asserted by hand. Each file's justification is derived:

  imported by X      a Python module some other module imports
  entry point        a script with a __main__ guard or CLI
  cited in DOC       a path referenced from PROVENANCE.md, REPRODUCING.md,
                     README.md, or data/README.md
  test               under tests/
  data               benchmark input
  ORPHAN             nothing references it

ORPHAN rows are the point. A file that maps to no paper claim, no import, and
no document has no reason to be in a reviewer-facing repository.

Run from the repository root:
    python scripts/build_manifest.py
"""

import ast
import fnmatch
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

DOCS = ["PROVENANCE.md", "REPRODUCING.md", "README.md", "data/README.md"]
GROUPED = {"data/cohen/pubmed_cache/": "benchmark cache, one JSON per PMID",
           "data/cohen/cache/": "working cache, gitignored"}


def tracked():
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                         check=True).stdout.splitlines()
    return [p for p in out if p.strip()]


def group(path):
    for prefix in GROUPED:
        if path.startswith(prefix):
            return prefix
    return None


def module_name(path):
    """src/auto_mesh.py -> {'auto_mesh', 'src.auto_mesh'}"""
    p = Path(path)
    if p.suffix != ".py":
        return set()
    stem = p.stem
    dotted = str(p.with_suffix("")).replace("/", ".").replace("\\", ".")
    return {stem, dotted}


def imports_of(path):
    """Every module name a Python file imports, absolute or relative."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return set()
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                found.add(a.name.split(".")[0])
                found.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.add(node.module.split(".")[-1])
                found.add(node.module)
            for a in node.names:
                found.add(a.name)
    return found


def has_main(path):
    try:
        t = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return '__name__ == "__main__"' in t or "__name__ == '__main__'" in t


def doc_references():
    """Every path-like token appearing in backticks in the documentation."""
    refs = defaultdict(set)
    for doc in DOCS:
        p = Path(doc)
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        fenced = " ".join(re.findall(r"```[^`]*```", text, re.S))
        tokens = re.findall(r"`([^`\n]+)`", text)
        tokens += re.findall(r"[\w./-]+\.(?:py|sh|json|csv|tsv|md|ipynb|toml|txt|pdf|png)",
                             fenced)
        for token in tokens:
            token = token.strip().split()[0].rstrip(".,;:")
            if "/" in token or token.endswith((".py", ".sh", ".json",
                                               ".csv", ".tsv", ".md",
                                               ".ipynb", ".toml", ".txt")):
                refs[token].add(doc)
    return refs


def main():
    files = tracked()
    if not files:
        sys.exit("no tracked files; run from the repository root")

    py = [f for f in files if f.endswith(".py")]
    provides = {}
    for f in py:
        for name in module_name(f):
            provides[name] = f

    imported_by = defaultdict(set)
    for f in py:
        for name in imports_of(f):
            target = provides.get(name)
            if target and target != f:
                imported_by[target].add(f)

    refs = doc_references()

    def justify(f):
        reasons = []
        if f in imported_by:
            users = sorted(imported_by[f])
            shown = ", ".join(Path(u).name for u in users[:3])
            if len(users) > 3:
                shown += f", +{len(users) - 3} more"
            reasons.append(f"imported by {shown}")
        if f.endswith(".py") and has_main(f):
            reasons.append("entry point")
        fname = Path(f).name
        hits = set()
        for token, ds in refs.items():
            pat = re.sub(r"\{[^}]*\}", "*", token)
            if (token == f or token.endswith(fname)
                    or fnmatch.fnmatch(f, "*" + pat)
                    or fnmatch.fnmatch(fname, pat)):
                hits |= ds
        if hits:
            reasons.append("cited in " + ", ".join(sorted(hits)))
        if f.startswith("tests/"):
            reasons.append("test")
        if f.startswith("data/") and not f.endswith(".md"):
            reasons.append("data")
        if f in DOCS or f.endswith(("LICENSE", ".gitignore", "pyproject.toml",
                                    "requirements.txt", "__init__.py",
                                    "README.md")):
            reasons.append("project file")
        if f.endswith(".png") and (Path(f).with_suffix(".pdf").as_posix() in
                                   {tok for tok in refs}
                                   or any(t2.endswith(Path(f).stem + ".pdf")
                                          for t2 in refs)):
            reasons.append("figure, PDF sibling cited")
        return "; ".join(reasons) if reasons else "**ORPHAN**"

    rows, seen_groups, orphans = [], {}, []
    for f in files:
        g = group(f)
        if g:
            seen_groups[g] = seen_groups.get(g, 0) + 1
            continue
        why = justify(f)
        if why == "**ORPHAN**":
            orphans.append(f)
        rows.append((f, why))

    out = ["# MANIFEST",
           "",
           "Generated by `scripts/build_manifest.py`. Do not edit by hand.",
           "",
           "Each tracked file with the reason it is in the repository, derived",
           "from Python imports and from paths cited in the documentation.",
           "**ORPHAN** means nothing references the file: no import, no paper",
           "claim, no document. Orphans should be removed or documented.",
           ""]

    if orphans:
        out += [f"## {len(orphans)} orphans", ""]
        out += [f"- `{o}`" for o in orphans] + [""]
    else:
        out += ["## No orphans", ""]

    out += ["## Grouped", "",
            "| Path | Files | Role |", "|---|---|---|"]
    for g, n in sorted(seen_groups.items()):
        out.append(f"| `{g}` | {n} | {GROUPED[g]} |")

    out += ["", "## Files", "", "| File | Why it is here |", "|---|---|"]
    for f, why in rows:
        out.append(f"| `{f}` | {why} |")
    out.append("")

    Path("MANIFEST.md").write_text("\n".join(out), encoding="utf-8")
    print(f"MANIFEST.md written: {len(rows)} files, {len(orphans)} orphans")
    for o in orphans:
        print(f"  ORPHAN {o}")
    return 1 if orphans else 0


if __name__ == "__main__":
    sys.exit(main())
