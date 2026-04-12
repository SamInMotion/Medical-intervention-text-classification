"""Text preprocessing: tokenization, n-grams, NEO enrichment, stopwords.

The ontology enrichment is the thesis contribution. For each token
that appears in the NEO dictionary, its synonyms and/or parent
concepts get appended to the document's token list. This gives the
classifier semantic context that a plain bag-of-words misses on
small datasets.
"""

import re
from typing import List, Optional, Set

from .config import WorkflowConfig


def word_tokens(text: str) -> List[str]:
    """Tokenize into lowercase words, preserving hyphenated compounds."""
    return re.findall(r"[\w-]+", text.casefold())


def make_ngrams(tokens: List[str], n: int = 3) -> List[str]:
    return ["_".join(tpl) for tpl in zip(*[tokens[i:] for i in range(n)])]


def _try_synonyms(word, neo_dict):
    val = neo_dict.get(word)
    if val:
        return val.get("Synonyms")
    return None


def _try_parents(word, neo_dict):
    val = neo_dict.get(word)
    if val:
        return val.get("Parents")
    return None


def _split_terms(terms, use_ngrams):
    """When n-grams are off, split multi-word terms like 'cerebellar_signs'
    into individual words so they match unigram features. When n-grams are
    on, keep them joined so they match n-gram features instead."""
    if not use_ngrams:
        return {s for sublist in [t.split("_") for t in terms] for s in sublist}
    return set(terms)


def _get_all_synonyms(tokens, neo_dict, use_ngrams):
    new = set()
    for w in tokens:
        synonyms = _try_synonyms(w, neo_dict)
        if synonyms:
            new |= _split_terms(synonyms, use_ngrams)
    return new


def _get_all_parents(tokens, neo_dict, use_ngrams):
    new = set()
    for w in tokens:
        parents = _try_parents(w, neo_dict)
        if parents:
            new |= _split_terms(parents, use_ngrams)
    return new


def enrich_with_neo(token_list, neo_dict, config, neo_terms_added=None):
    """Add NEO synonyms and/or parent concepts to a document's tokens."""
    if not config.synonyms and not config.parents:
        return token_list

    use_ngrams = bool(config.ngrams)
    token_set = set(token_list)
    new = set()

    if config.synonyms:
        new |= _get_all_synonyms(token_set, neo_dict, use_ngrams)

    if config.parents:
        new |= _get_all_parents(token_set, neo_dict, use_ngrams)

    if new:
        new -= token_set
        if neo_terms_added is not None:
            neo_terms_added |= new
        token_list = list(token_set | new)

    return token_list


def remove_stopwords(token_list, stopwords):
    return [tok for tok in token_list if tok not in stopwords]


def preprocess_corpus(texts, config, neo_dict=None, stopwords=None):
    """Run the full preprocessing pipeline on raw texts.

    Order matters: tokenize -> n-grams -> NEO enrichment -> stopwords.
    N-grams come before enrichment so that multi-word ontology terms
    can match n-gram features when that mode is active.
    """
    tokenized = [word_tokens(t) for t in texts]

    if config.ngrams:
        n = int(config.ngrams)
        for txt in tokenized:
            txt.extend(make_ngrams(txt, n))

    neo_terms_added: Set[str] = set()
    if neo_dict and (config.synonyms or config.parents):
        tokenized = [
            enrich_with_neo(t, neo_dict, config, neo_terms_added)
            for t in tokenized
        ]

    if config.dropstop and stopwords:
        tokenized = [remove_stopwords(t, stopwords) for t in tokenized]

    return tokenized
