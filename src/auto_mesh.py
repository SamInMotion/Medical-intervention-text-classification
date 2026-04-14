"""Automatic MeSH term lookup for abstracts.

This module builds a MeSH vocabulary from cached PubMed records and
matches terms against abstract text by string search. This is the
fair comparison to NEO enrichment: same method (mechanical lookup),
different vocabulary (MeSH instead of NEO).

The key difference from expert-assigned MeSH:
- Expert-assigned: a human indexer reads the article and selects
  the most relevant MeSH terms for THAT specific article
- Auto-lookup: a string matching algorithm checks whether any
  MeSH term from the full vocabulary appears in the abstract text

If auto-lookup helps as much as expert-assigned, the benefit comes
from the vocabulary itself. If auto-lookup doesn't help (like NEO
didn't help), the benefit comes from expert contextual curation.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


def build_mesh_vocabulary(cache_dir: str, min_length: int = 4) -> Set[str]:
    """Build a set of all unique MeSH terms from cached PubMed records.

    Args:
        cache_dir: directory containing per-PMID JSON cache files
        min_length: minimum term length to include (filters out
            MeSH qualifiers like "use" that would match everywhere)

    Returns:
        set of lowercase MeSH term strings
    """
    cache_path = Path(cache_dir)
    vocab = set()
    file_count = 0

    for json_file in cache_path.glob("*.json"):
        with open(json_file, "r") as f:
            record = json.load(f)
        for term in record.get("mesh_terms", []):
            if len(term) >= min_length:
                vocab.add(term.lower())
        file_count += 1

    logger.info(
        "Built MeSH vocabulary: %d unique terms from %d records",
        len(vocab), file_count,
    )
    return vocab


def lookup_mesh_in_text(
    text: str,
    mesh_vocab: Set[str],
) -> List[str]:
    """Find MeSH terms that appear as substrings in the text.

    Case-insensitive matching. Returns the matched terms in their
    original (lowercase) form from the vocabulary.

    This is the automatic analog of expert MeSH assignment:
    mechanical string matching without contextual judgment.
    """
    text_lower = text.lower()
    matched = []
    for term in mesh_vocab:
        if term in text_lower:
            matched.append(term)
    return matched


def prepare_auto_mesh_texts(
    df,
    mesh_vocab: Set[str],
) -> List[str]:
    """Build text with auto-looked-up MeSH terms appended.

    For each abstract, scans the text for MeSH vocabulary matches
    and appends them. Uses abstract text only (not title) to keep
    the lookup analogous to NEO enrichment on the thesis data.

    Args:
        df: DataFrame with 'texts' column (abstracts)
        mesh_vocab: set from build_mesh_vocabulary

    Returns:
        list of strings: abstract + matched MeSH terms
    """
    texts = []
    total_matches = 0

    for _, row in df.iterrows():
        abstract = str(row["texts"])
        matched = lookup_mesh_in_text(abstract, mesh_vocab)
        total_matches += len(matched)

        if matched:
            mesh_str = " ".join(matched)
            texts.append(f"{abstract} {mesh_str}")
        else:
            texts.append(abstract)

    avg_matches = total_matches / len(df) if len(df) > 0 else 0
    logger.info(
        "Auto MeSH lookup: %.1f terms matched per abstract (avg), "
        "%d total matches across %d abstracts",
        avg_matches, total_matches, len(df),
    )
    return texts
