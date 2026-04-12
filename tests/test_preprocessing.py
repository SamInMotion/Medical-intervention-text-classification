"""Tests for preprocessing functions."""

import pytest
from src.config import WorkflowConfig
from src.preprocessing import (
    word_tokens,
    make_ngrams,
    enrich_with_neo,
    remove_stopwords,
    preprocess_corpus,
)


class TestWordTokens:
    def test_basic(self):
        assert word_tokens("Hello World") == ["hello", "world"]

    def test_casefold(self):
        result = word_tokens("Background: ALZHEIMER'S Disease")
        assert "background" in result
        assert "alzheimer" in result
        assert "disease" in result

    def test_hyphenated(self):
        result = word_tokens("evidence-based follow-up")
        assert "evidence-based" in result
        assert "follow-up" in result

    def test_numbers(self):
        result = word_tokens("95 confidence interval p 0.05")
        assert "95" in result
        assert "0" in result
        assert "05" in result

    def test_empty(self):
        assert word_tokens("") == []


class TestMakeNgrams:
    def test_trigrams(self):
        result = make_ngrams(["a", "b", "c", "d", "e"], 3)
        assert result == ["a_b_c", "b_c_d", "c_d_e"]

    def test_bigrams(self):
        result = make_ngrams(["a", "b", "c"], 2)
        assert result == ["a_b", "b_c"]

    def test_too_short(self):
        assert make_ngrams(["a", "b"], 3) == []

    def test_single_token(self):
        assert make_ngrams(["a"], 3) == []


@pytest.fixture
def sample_neo_dict():
    return {
        "ataxia": {
            "Synonyms": ["dyssynergia", "dystaxia"],
            "Parents": ["cerebellar_signs"],
        },
        "dystaxia": {
            "Synonyms": [],
            "Parents": ["ataxia"],
        },
        "dementia": {
            "Synonyms": [],
            "Parents": ["impaired_cognition"],
        },
        "agitated": {
            "Synonyms": ["agitation"],
            "Parents": ["abnormal_affect"],
        },
        "agitation": {
            "Synonyms": ["agitated"],
            "Parents": [],
        },
    }


class TestEnrichWithNeo:
    def test_synonyms_added(self, sample_neo_dict):
        config = WorkflowConfig(synonyms=True, parents=False, ngrams=False)
        tokens = ["the", "patient", "has", "ataxia"]
        result = set(enrich_with_neo(tokens, sample_neo_dict, config))
        assert "dyssynergia" in result
        assert "dystaxia" in result

    def test_parents_added(self, sample_neo_dict):
        config = WorkflowConfig(synonyms=False, parents=True, ngrams=False)
        tokens = ["ataxia"]
        result = set(enrich_with_neo(tokens, sample_neo_dict, config))
        # without ngrams, multi-word parents get split
        assert "cerebellar" in result or "cerebellar_signs" in result

    def test_no_enrichment_when_disabled(self, sample_neo_dict):
        config = WorkflowConfig(synonyms=False, parents=False)
        tokens = ["the", "patient", "has", "ataxia"]
        result = enrich_with_neo(tokens, sample_neo_dict, config)
        assert result == tokens

    def test_no_duplicates(self, sample_neo_dict):
        config = WorkflowConfig(synonyms=True, parents=False, ngrams=False)
        # agitated and agitation are mutual synonyms
        tokens = ["agitated", "agitation"]
        result = enrich_with_neo(tokens, sample_neo_dict, config)
        assert len(result) == len(set(result))

    def test_tracking_set(self, sample_neo_dict):
        config = WorkflowConfig(synonyms=True, parents=True, ngrams=False)
        tracker = set()
        enrich_with_neo(["dementia"], sample_neo_dict, config, neo_terms_added=tracker)
        assert len(tracker) > 0


class TestRemoveStopwords:
    def test_basic(self):
        result = remove_stopwords(["the", "patient", "has", "ataxia"], {"the", "has"})
        assert result == ["patient", "ataxia"]

    def test_empty_stopwords(self):
        tokens = ["the", "patient"]
        assert remove_stopwords(tokens, set()) == tokens

    def test_all_removed(self):
        assert remove_stopwords(["the", "a", "is"], {"the", "a", "is"}) == []


class TestPreprocessCorpus:
    def test_basic(self):
        config = WorkflowConfig()
        result = preprocess_corpus(["The patient has dementia."], config)
        assert len(result) == 1
        assert "the" in result[0]
        assert "patient" in result[0]
        assert "dementia" in result[0]

    def test_with_stopwords(self):
        config = WorkflowConfig(dropstop=True)
        result = preprocess_corpus(
            ["The patient has dementia."], config, stopwords={"the", "has"}
        )
        assert "the" not in result[0]
        assert "has" not in result[0]
        assert "patient" in result[0]

    def test_with_ngrams(self):
        config = WorkflowConfig(ngrams=3)
        result = preprocess_corpus(["a b c d"], config)
        assert "a_b_c" in result[0]
        assert "b_c_d" in result[0]
