"""Amend paper_draft_v5.tex and Consolidation_v4.md with token-length audit findings.

Three additions to the paper:
  1. §3.3 (BERT classifier): numerical disclosure of truncation rates per topic-mode
  2. §5.1 (Discussion): paragraph acknowledging truncation as a third contributing factor
                        to the BoW-BERT difference at Statins
  3. §6 (Limitations): paragraph framing the asymmetry as known and disclosed rather than
                       confounding

One addition to the consolidation:
  4. New §6.5 token-length audit summary

Idempotent. Creates .bak before modifying. Run from repo root.

Usage:
    python paper_experiments/patch_v5_truncation.py
"""

from pathlib import Path
import shutil
import sys


PAPER_PATH = Path("paper/paper_draft_v5.tex")
CONSOLIDATION_PATH = Path("docs/Cohen_BERT_Extension_Results_Consolidation_v4.md")

# ----------------------------------------------------------------------------
# Paper edit 1: §3.3 BERT classifier — after the max_length=512 sentence
# ----------------------------------------------------------------------------
P1_ANCHOR = "maximum sequence length 512 tokens. Class weights followed sklearn's balanced formulation"
P1_REPLACE = (
    "maximum sequence length 512 tokens. "
    "Token-length analysis of the input corpus shows that 4.6--7.9\\% of "
    "abstracts across the three topics exceed this 512-token cap in the "
    "\\texttt{abstract} and \\texttt{title\\_abstract} modes and are truncated; "
    "the rate rises to 15.1\\% for Statins, 10.4\\% for Opioids, and 11.8\\% "
    "for ADHD in the \\texttt{title\\_abstract\\_mesh} mode where the MeSH "
    "terms appended at the end of the input are most affected by truncation. "
    "The bag-of-words pipeline is unbounded and sees the full input in every "
    "case. This representation-asymmetry between the two classifiers is "
    "discussed in Sections~5.1 and~6. "
    "Class weights followed sklearn's balanced formulation"
)

# ----------------------------------------------------------------------------
# Paper edit 2: §5.1 Discussion — paragraph after "could resolve the question."
# ----------------------------------------------------------------------------
P2_ANCHOR = (
    "Future work that varies per-fold training volume systematically (e.g.\\ via "
    "learning-curve analysis at varying training-set fractions) could resolve "
    "the question."
)
P2_REPLACE = (
    "Future work that varies per-fold training volume systematically (e.g.\\ via "
    "learning-curve analysis at varying training-set fractions) could resolve "
    "the question."
    "\n\n"
    "A third contributing factor to the BoW-BERT difference at Statins, "
    "distinct from training volume, is BERT's 512-token truncation. In the "
    "\\texttt{title\\_abstract\\_mesh} mode where the canonical bag-of-words "
    "gap arises, 15.1\\% of Statins inputs are truncated at 512 BiomedBERT "
    "subword tokens (Section~3.3), and the truncated content is preferentially "
    "the MeSH terms that are appended at the end of the input. The bag-of-words "
    "classifier sees the full MeSH augmentation in every case. The BiomedBERT "
    "Statins result of $+0.020$ could therefore reflect not only the "
    "transformer's algorithmic absorption of the assignment-mechanism "
    "difference but also reduced exposure to the augmentation content the "
    "difference is observable through. This contribution cannot be isolated "
    "from the training-volume contribution without a no-truncation BERT "
    "comparison (e.g.\\ Longformer or token-budget-restructured input ordering "
    "that places MeSH terms before the abstract body), which we leave as future "
    "work and disclose as a limitation in Section~6."
)

# ----------------------------------------------------------------------------
# Paper edit 3: §6 Limitations — paragraph after the transformer-comparison para
# ----------------------------------------------------------------------------
P3_ANCHOR = (
    "Whether the absorption finding generalises to BioBERT, BiomedBERT-large, "
    "BioLinkBERT, or to more recent biomedical large language models is an "
    "open question. The pattern's robustness across architectures is the "
    "natural next test."
)
P3_REPLACE = (
    "Whether the absorption finding generalises to BioBERT, BiomedBERT-large, "
    "BioLinkBERT, or to more recent biomedical large language models is an "
    "open question. The pattern's robustness across architectures is the "
    "natural next test."
    "\n\n"
    "The BiomedBERT 512-token sequence cap truncates a topic-dependent share "
    "of inputs: in the \\texttt{title\\_abstract\\_mesh} mode, 15.1\\% of "
    "Statins, 10.4\\% of Opioids, and 11.8\\% of ADHD inputs exceed the cap, "
    "and the truncated content is preferentially the MeSH terms appended at "
    "the end of the input. The bag-of-words pipeline is unbounded and sees "
    "the full input in every case. This representation-asymmetry could "
    "contribute to the smaller BiomedBERT Statins gap relative to the canonical "
    "bag-of-words reference; we cannot isolate this contribution from the "
    "training-volume contribution without a no-truncation BERT comparison "
    "(e.g.\\ Longformer architecture or token-budget-restructured input "
    "ordering that places MeSH terms before the abstract body). We disclose "
    "this asymmetry in Section~3.3 rather than treat it as a confound the "
    "present analysis can correct for."
)

PAPER_EDITS = [
    ("§3.3 BERT classifier — truncation rate disclosure", P1_ANCHOR, P1_REPLACE),
    ("§5.1 Discussion — truncation as third contributing factor", P2_ANCHOR, P2_REPLACE),
    ("§6 Limitations — representation-asymmetry paragraph", P3_ANCHOR, P3_REPLACE),
]

# ----------------------------------------------------------------------------
# Consolidation edit: add §6.5 token-length audit
# ----------------------------------------------------------------------------
C1_ANCHOR = "### §6.4 What we did NOT run"
C1_REPLACE = (
    "### §6.5 Token-length audit (added July 1, 2026)\n\n"
    "BiomedBERT tokenizer applied to the three topics' input texts under each "
    "of the three non-auto-mesh text modes. Truncation at 512 tokens disclosed "
    "in paper §3.3, §5.1, and §6.\n\n"
    "| Topic | Mode | n | Median tokens | Max | Truncated@512 |\n"
    "|---|---|---|---|---|---|\n"
    "| Statins | abstract | 2744 | 312 | 1929 | 167 (6.1%) |\n"
    "| Statins | title_abstract | 2744 | 331 | 1947 | 218 (7.9%) |\n"
    "| **Statins** | **title_abstract_mesh** | **2744** | **381** | **1982** | **415 (15.1%)** |\n"
    "| Opioids | abstract | 1772 | 291 | 1075 | 83 (4.7%) |\n"
    "| Opioids | title_abstract | 1772 | 308 | 1095 | 103 (5.8%) |\n"
    "| **Opioids** | **title_abstract_mesh** | **1772** | **357** | **1163** | **184 (10.4%)** |\n"
    "| ADHD | abstract | 803 | 294 | 986 | 37 (4.6%) |\n"
    "| ADHD | title_abstract | 803 | 313 | 1002 | 44 (5.5%) |\n"
    "| **ADHD** | **title_abstract_mesh** | **803** | **363** | **1046** | **95 (11.8%)** |\n\n"
    "Reading: BiomedBERT sees less of the MeSH augmentation precisely at the "
    "topic-mode combination where MeSH augmentation matters most for the bag-of-words "
    "classifier (Statins title_abstract_mesh). This adds a third candidate "
    "explanation alongside the training-volume and canonical-overestimate readings "
    "in §4. Disclosed in paper §3.3 + §5.1 + §6 rather than treated as a confound.\n\n"
    "Output files (under paper_experiments/outputs/):\n"
    "- `audit_token_lengths.json` — raw per-topic-mode statistics\n"
    "- `audit_token_lengths.md` — paper-quality table\n\n"
    "### §6.6 What we did NOT run"
)

# ----------------------------------------------------------------------------
# Apply
# ----------------------------------------------------------------------------
def fail(msg):
    print(f"[FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def apply_edits(path, edits, label):
    if not path.exists():
        fail(f"Target file not found: {path}")
    src = path.read_text(encoding="utf-8")
    already = 0
    will_apply = []
    for name, anchor, replace in edits:
        if anchor not in src:
            # Maybe already applied — check for a distinctive part of the replacement
            distinctive = replace.replace(anchor, "").strip().split("\n")[0]
            distinctive = distinctive[:80] if distinctive else ""
            if distinctive and distinctive in src:
                print(f"  [skip] {name} — already applied")
                already += 1
            else:
                fail(f"Anchor not found for: {name}\n"
                     f"  Looked for: {anchor[:100]}...\n"
                     f"  And could not detect prior application.")
        else:
            will_apply.append((name, anchor, replace))

    if not will_apply:
        print(f"  [no-op] {label} already fully patched ({already}/{len(edits)})")
        return False

    backup = path.with_suffix(path.suffix + ".bak2")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  [backup] {backup}")

    for name, anchor, replace in will_apply:
        src = src.replace(anchor, replace, 1)
        print(f"  [applied] {name}")

    path.write_text(src, encoding="utf-8")
    print(f"  [written] {path}")
    return True


def main():
    print("Patching paper_draft_v5.tex...")
    apply_edits(PAPER_PATH, PAPER_EDITS, "paper_draft_v5.tex")
    print()

    print("Patching Cohen_BERT_Extension_Results_Consolidation_v4.md...")
    apply_edits(
        CONSOLIDATION_PATH,
        [("§6.5 token-length audit", C1_ANCHOR, C1_REPLACE)],
        "Consolidation_v4.md",
    )
    print()

    print("Done. Next:")
    print("  git add paper/paper_draft_v5.tex docs/Cohen_BERT_Extension_Results_Consolidation_v4.md \\")
    print("          paper_experiments/outputs/audit_token_lengths.json \\")
    print("          paper_experiments/outputs/audit_token_lengths.md")
    print("  git commit -m 'Add BERT token-length audit (§3.3, §5.1, §6, consolidation §6.5)'")
    print("  git pull --rebase origin main  # because last push was rejected")
    print("  git push origin main")
    print("  cp paper/paper_draft_v5.tex docs/Cohen_BERT_Extension_Results_Consolidation_v4.md \\")
    print("     paper_experiments/outputs/audit_token_lengths.{json,md} \\")
    print("     '/g/My Drive/cohen_bert_run/'")


if __name__ == "__main__":
    main()
