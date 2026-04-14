"""Tests for Cohen pipeline text preparation logic."""

import pandas as pd
import pytest

from src.cohen_pipeline import _prepare_texts


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            "pmid": "111",
            "labels": 1,
            "texts": "This study evaluated statin therapy.",
            "title": "Statin outcomes in elderly patients",
            "mesh_terms": ["Statins", "Aged", "Treatment Outcome"],
        },
        {
            "pmid": "222",
            "labels": 0,
            "texts": "Diet modification was compared to drug therapy.",
            "title": "Dietary interventions review",
            "mesh_terms": ["Diet", "Drug Therapy"],
        },
    ])


def test_abstract_only(sample_df):
    texts = _prepare_texts(sample_df, "abstract")
    assert texts[0] == "This study evaluated statin therapy."
    assert "Statin outcomes" not in texts[0]


def test_title_abstract(sample_df):
    texts = _prepare_texts(sample_df, "title_abstract")
    assert texts[0].startswith("Statin outcomes in elderly patients")
    assert "statin therapy" in texts[0]


def test_title_abstract_mesh(sample_df):
    texts = _prepare_texts(sample_df, "title_abstract_mesh")
    assert "Statin outcomes" in texts[0]
    assert "statin therapy" in texts[0]
    assert "Treatment Outcome" in texts[0]
    assert "Aged" in texts[0]


def test_mesh_not_in_abstract_mode(sample_df):
    texts = _prepare_texts(sample_df, "abstract")
    assert "Treatment Outcome" not in texts[0]
    assert "Aged" not in texts[0]


def test_empty_mesh_handled():
    df = pd.DataFrame([{
        "pmid": "333",
        "labels": 1,
        "texts": "Some abstract text.",
        "title": "Some title",
        "mesh_terms": [],
    }])
    texts = _prepare_texts(df, "title_abstract_mesh")
    assert texts[0] == "Some title Some abstract text."


def test_missing_title_handled():
    df = pd.DataFrame([{
        "pmid": "444",
        "labels": 0,
        "texts": "Abstract without title.",
        "title": "",
        "mesh_terms": ["Term1"],
    }])
    texts = _prepare_texts(df, "title_abstract")
    # empty title should not leave leading space
    assert texts[0] == "Abstract without title."
