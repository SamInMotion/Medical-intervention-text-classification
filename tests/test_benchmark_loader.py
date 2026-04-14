"""Tests for Cohen benchmark loader and WSS@95% metric."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.benchmark_loader import (
    COHEN_TOPICS,
    parse_cohen_tsv,
    get_topic_data,
    build_topic_dataframe,
    list_topics_summary,
    _parse_pubmed_xml_article,
)
from src.evaluation import compute_wss_at_recall


# --- fixtures ---

@pytest.fixture
def sample_tsv(tmp_path):
    """Create a minimal Cohen-format TSV for testing."""
    lines = [
        "Statins\t100\t11111111\tI\tI\r\n",
        "Statins\t101\t22222222\tE\tE\r\n",
        "Statins\t102\t33333333\t3\tE\r\n",
        "Statins\t103\t44444444\tI\tE\r\n",
        "ADHD\t200\t55555555\tI\tI\r\n",
        "ADHD\t201\t66666666\tE\tE\r\n",
        "ADHD\t202\t77777777\t5\t5\r\n",
    ]
    tsv_file = tmp_path / "test_cohen.tsv"
    with open(tsv_file, "w", newline="") as f:
        f.writelines(lines)
    return str(tsv_file)


@pytest.fixture
def sample_records():
    """Simulated PubMed fetch results."""
    return {
        "11111111": {
            "pmid": "11111111",
            "title": "Statin therapy outcomes",
            "abstract": "This study evaluated statin therapy in patients with high cholesterol.",
            "mesh_terms": ["Hydroxymethylglutaryl-CoA Reductase Inhibitors", "Cholesterol"],
        },
        "22222222": {
            "pmid": "22222222",
            "title": "Dietary interventions",
            "abstract": "Dietary changes were compared to pharmacological treatment.",
            "mesh_terms": ["Diet", "Drug Therapy"],
        },
        "33333333": {
            "pmid": "33333333",
            "title": "Wrong drug class",
            "abstract": "Beta blocker efficacy in hypertension management.",
            "mesh_terms": ["Adrenergic beta-Antagonists"],
        },
        "44444444": {
            "pmid": "44444444",
            "title": "Statin safety profile",
            "abstract": "Long-term safety of statin use was assessed.",
            "mesh_terms": ["Drug-Related Side Effects and Adverse Reactions"],
        },
    }


@pytest.fixture
def sample_xml_article():
    """Minimal PubMed XML for one article."""
    import xml.etree.ElementTree as ET
    xml_str = """
    <PubmedArticle>
        <MedlineCitation>
            <PMID>99999999</PMID>
            <Article>
                <ArticleTitle>Test article about drug efficacy</ArticleTitle>
                <Abstract>
                    <AbstractText Label="OBJECTIVE">To test drug efficacy.</AbstractText>
                    <AbstractText Label="RESULTS">The drug was effective.</AbstractText>
                </Abstract>
            </Article>
            <MeshHeadingList>
                <MeshHeading>
                    <DescriptorName>Drug Therapy</DescriptorName>
                    <QualifierName>adverse effects</QualifierName>
                </MeshHeading>
                <MeshHeading>
                    <DescriptorName>Treatment Outcome</DescriptorName>
                </MeshHeading>
            </MeshHeadingList>
        </MedlineCitation>
    </PubmedArticle>
    """
    return ET.fromstring(xml_str)


# --- parse_cohen_tsv ---

def test_parse_tsv_row_count(sample_tsv):
    df = parse_cohen_tsv(sample_tsv)
    assert len(df) == 7


def test_parse_tsv_columns(sample_tsv):
    df = parse_cohen_tsv(sample_tsv)
    expected = {"topic", "endnote_id", "pmid", "abstract_label",
                "article_label", "abstract_decision_raw", "article_decision_raw"}
    assert set(df.columns) == expected


def test_parse_tsv_label_mapping(sample_tsv):
    df = parse_cohen_tsv(sample_tsv)
    statins = df[df["topic"] == "Statins"]
    # row 0: I -> 1, row 1: E -> 0, row 2: 3 -> 0, row 3: I -> 1
    assert statins["abstract_label"].tolist() == [1, 0, 0, 1]


def test_parse_tsv_strips_carriage_return(sample_tsv):
    """The real file has \\r\\n endings. Verify no \\r in parsed values."""
    df = parse_cohen_tsv(sample_tsv)
    for col in df.columns:
        for val in df[col]:
            assert "\r" not in str(val)


def test_parse_tsv_topics(sample_tsv):
    df = parse_cohen_tsv(sample_tsv)
    assert set(df["topic"].unique()) == {"Statins", "ADHD"}


# --- get_topic_data ---

def test_get_topic_data_statins(sample_tsv):
    df = parse_cohen_tsv(sample_tsv)
    pmids, labels = get_topic_data(df, "Statins")
    assert len(pmids) == 4
    assert labels == [1, 0, 0, 1]


def test_get_topic_data_article_level(sample_tsv):
    df = parse_cohen_tsv(sample_tsv)
    pmids, labels = get_topic_data(df, "Statins", level="article")
    # I, E, E, E at article level
    assert labels == [1, 0, 0, 0]


def test_get_topic_data_invalid_topic(sample_tsv):
    df = parse_cohen_tsv(sample_tsv)
    with pytest.raises(ValueError, match="Unknown topic"):
        get_topic_data(df, "FakeTopic")


# --- build_topic_dataframe ---

def test_build_dataframe_shape(sample_records):
    pmids = ["11111111", "22222222", "33333333", "44444444"]
    labels = [1, 0, 0, 1]
    df = build_topic_dataframe(pmids, labels, sample_records)
    assert len(df) == 4
    assert "labels" in df.columns
    assert "texts" in df.columns
    assert "mesh_terms" in df.columns


def test_build_dataframe_drops_missing():
    pmids = ["11111111", "99999999"]
    labels = [1, 0]
    records = {
        "11111111": {
            "pmid": "11111111", "title": "T",
            "abstract": "Some text", "mesh_terms": [],
        },
    }
    df = build_topic_dataframe(pmids, labels, records)
    assert len(df) == 1
    assert df.iloc[0]["pmid"] == "11111111"


def test_build_dataframe_drops_empty_abstract():
    pmids = ["11111111"]
    labels = [1]
    records = {
        "11111111": {
            "pmid": "11111111", "title": "T",
            "abstract": "", "mesh_terms": [],
        },
    }
    df = build_topic_dataframe(pmids, labels, records)
    assert len(df) == 0


# --- XML parsing ---

def test_parse_xml_pmid(sample_xml_article):
    record = _parse_pubmed_xml_article(sample_xml_article)
    assert record["pmid"] == "99999999"


def test_parse_xml_structured_abstract(sample_xml_article):
    record = _parse_pubmed_xml_article(sample_xml_article)
    assert "OBJECTIVE:" in record["abstract"]
    assert "RESULTS:" in record["abstract"]
    assert "drug was effective" in record["abstract"]


def test_parse_xml_mesh_terms(sample_xml_article):
    record = _parse_pubmed_xml_article(sample_xml_article)
    assert "Drug Therapy" in record["mesh_terms"]
    assert "Treatment Outcome" in record["mesh_terms"]
    assert "adverse effects" in record["mesh_terms"]


def test_parse_xml_title(sample_xml_article):
    record = _parse_pubmed_xml_article(sample_xml_article)
    assert "drug efficacy" in record["title"]


# --- PubMed caching ---

def test_cache_write_and_read(tmp_path, sample_records):
    """Verify that cached JSON files can be read back correctly."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # simulate what fetch_pubmed_records does when caching
    for pmid, record in sample_records.items():
        with open(cache_dir / f"{pmid}.json", "w") as f:
            json.dump(record, f)

    # read them back
    for pmid, expected in sample_records.items():
        with open(cache_dir / f"{pmid}.json", "r") as f:
            loaded = json.load(f)
        assert loaded["pmid"] == expected["pmid"]
        assert loaded["abstract"] == expected["abstract"]
        assert loaded["mesh_terms"] == expected["mesh_terms"]


# --- list_topics_summary ---

def test_topics_summary(sample_tsv):
    summary = list_topics_summary(sample_tsv)
    assert len(summary) == 2  # only Statins and ADHD in fixture
    statins_row = summary[summary["topic"] == "Statins"].iloc[0]
    assert statins_row["total"] == 4
    assert statins_row["included_abstract"] == 2


# --- WSS@95% ---

def test_wss_perfect_classifier():
    """A perfect classifier should save nearly all work."""
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    # perfect ranking: positives get highest probabilities
    y_proba = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    result = compute_wss_at_recall(y_true, y_proba, target_recall=0.95)
    # with 2 positives, need both (recall=1.0 >= 0.95)
    # review 2 docs, skip 8, baseline = 0.05
    # WSS = 8/10 - 0.05 = 0.75
    assert result["wss"] == pytest.approx(0.75, abs=0.01)
    assert result["achieved_recall"] >= 0.95


def test_wss_random_classifier():
    """A random classifier should have WSS near 0."""
    rng = np.random.RandomState(42)
    n = 1000
    y_true = np.zeros(n, dtype=int)
    y_true[:50] = 1  # 5% inclusion rate
    y_proba = rng.random(n)
    result = compute_wss_at_recall(y_true, y_proba, target_recall=0.95)
    # WSS should be close to 0 for random
    assert abs(result["wss"]) < 0.10


def test_wss_no_positives():
    y_true = np.array([0, 0, 0])
    y_proba = np.array([0.5, 0.3, 0.1])
    result = compute_wss_at_recall(y_true, y_proba)
    assert np.isnan(result["wss"])


def test_wss_all_positives():
    y_true = np.array([1, 1, 1])
    y_proba = np.array([0.9, 0.8, 0.7])
    result = compute_wss_at_recall(y_true, y_proba, target_recall=0.95)
    # need to review all 3 to get recall >= 0.95
    # skip 0 docs, WSS = 0/3 - 0.05 = -0.05
    assert result["wss"] == pytest.approx(-0.05, abs=0.01)


def test_wss_returns_dict_keys():
    y_true = np.array([1, 0, 0, 0])
    y_proba = np.array([0.9, 0.1, 0.2, 0.3])
    result = compute_wss_at_recall(y_true, y_proba)
    assert "wss" in result
    assert "threshold" in result
    assert "achieved_recall" in result
    assert "tn" in result
    assert "fn" in result
    assert "n" in result
