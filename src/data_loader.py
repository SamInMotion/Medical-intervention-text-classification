"""Data loading for abstracts, NEO ontology, and stopwords."""

import json
from typing import Tuple

import nltk
import pandas as pd

from .config import ProjectPaths


def load_abstracts(paths: ProjectPaths) -> pd.DataFrame:
    """Load the annotated abstracts dataset from TSV."""
    filepath = paths.abstracts_path
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            f"See data/README.md for instructions."
        )
    df = pd.read_csv(filepath, sep="\t")

    expected_cols = {"labels", "texts"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"Expected columns {expected_cols}, got {set(df.columns)}"
        )
    return df


def load_neo_ontology(paths: ProjectPaths) -> dict:
    """Load NEO ontology dictionary from JSON."""
    filepath = paths.neo_path
    if not filepath.exists():
        raise FileNotFoundError(
            f"NEO ontology not found at {filepath}. "
            f"See data/README.md for instructions."
        )
    with open(filepath, "r") as f:
        return json.load(f)


def load_stopwords(paths: ProjectPaths) -> set:
    """Merge NLTK English stopwords with domain-specific medical stopwords."""
    nltk.download("stopwords", quiet=True)
    eng_stopwords = set(nltk.corpus.stopwords.words("english"))

    med_stopwords = set()
    filepath = paths.med_stopwords_path
    if filepath.exists():
        with open(filepath, "r") as f:
            med_stopwords = set(f.read().splitlines())

    return eng_stopwords | med_stopwords


def get_labels_and_texts(df: pd.DataFrame) -> Tuple[pd.Series, list]:
    return df["labels"], df["texts"].tolist()
