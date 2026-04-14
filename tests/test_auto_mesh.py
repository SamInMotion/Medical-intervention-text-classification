"""Tests for automatic MeSH term lookup."""

import json
import pandas as pd
import pytest

from src.auto_mesh import (
    build_mesh_vocabulary,
    lookup_mesh_in_text,
    prepare_auto_mesh_texts,
)


@pytest.fixture
def mesh_cache(tmp_path):
    """Create cached records with MeSH terms."""
    records = [
        {"pmid": "111", "title": "T", "abstract": "A",
         "mesh_terms": ["Statins", "Cholesterol", "Drug Therapy", "Aged"]},
        {"pmid": "222", "title": "T", "abstract": "A",
         "mesh_terms": ["Statins", "Liver Diseases", "Drug Therapy"]},
        {"pmid": "333", "title": "T", "abstract": "A",
         "mesh_terms": ["Hypertension", "ACE"]},
    ]
    for rec in records:
        with open(tmp_path / f"{rec['pmid']}.json", "w") as f:
            json.dump(rec, f)
    return str(tmp_path)


def test_build_vocabulary_count(mesh_cache):
    vocab = build_mesh_vocabulary(mesh_cache)
    # statins, cholesterol, drug therapy, aged, liver diseases, hypertension
    # ACE is only 3 chars, filtered by min_length=4
    assert "ace" not in vocab
    assert "statins" in vocab
    assert "drug therapy" in vocab
    assert len(vocab) == 6


def test_build_vocabulary_lowercase(mesh_cache):
    vocab = build_mesh_vocabulary(mesh_cache)
    for term in vocab:
        assert term == term.lower()


def test_lookup_finds_matches():
    vocab = {"statins", "cholesterol", "drug therapy", "liver diseases"}
    text = "This study evaluated statins for cholesterol reduction."
    matched = lookup_mesh_in_text(text, vocab)
    assert "statins" in matched
    assert "cholesterol" in matched
    assert "drug therapy" not in matched  # not in text


def test_lookup_case_insensitive():
    vocab = {"hydroxymethylglutaryl-coa reductase inhibitors"}
    text = "Hydroxymethylglutaryl-CoA Reductase Inhibitors were studied."
    matched = lookup_mesh_in_text(text, vocab)
    assert len(matched) == 1


def test_lookup_no_matches():
    vocab = {"liver diseases", "hypertension"}
    text = "This study examined statin efficacy in elderly patients."
    matched = lookup_mesh_in_text(text, vocab)
    assert len(matched) == 0


def test_prepare_auto_mesh_texts():
    vocab = {"statins", "cholesterol"}
    df = pd.DataFrame([
        {"texts": "This study evaluated statins and cholesterol.", "labels": 1},
        {"texts": "Dietary changes in elderly patients.", "labels": 0},
    ])
    texts = prepare_auto_mesh_texts(df, vocab)
    # first abstract matches both terms
    assert "statins" in texts[0]
    assert "cholesterol" in texts[0]
    # second abstract matches neither - unchanged
    assert texts[1] == "Dietary changes in elderly patients."


def test_prepare_auto_mesh_appends_not_replaces():
    vocab = {"statins"}
    df = pd.DataFrame([
        {"texts": "Original abstract about statins.", "labels": 1},
    ])
    texts = prepare_auto_mesh_texts(df, vocab)
    assert texts[0].startswith("Original abstract about statins.")
    # the matched term is appended
    assert texts[0].count("statins") >= 2
