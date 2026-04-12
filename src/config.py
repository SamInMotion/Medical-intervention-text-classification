"""Workflow configurations and project paths.

Each workflow combines preprocessing options (stopword removal,
NEO synonym/parent enrichment, n-gram size, vocabulary cap) to
test how ontology enrichment affects classification.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union


@dataclass
class WorkflowConfig:
    dropstop: bool = False
    synonyms: bool = False
    parents: bool = False
    ngrams: Union[bool, int] = False
    nrfeats: int = 1000


# the 11 configurations from the thesis
WORKFLOWS = {
    0: WorkflowConfig(dropstop=False, synonyms=False, parents=False, ngrams=False, nrfeats=1000),
    1: WorkflowConfig(dropstop=True, synonyms=False, parents=False, ngrams=False, nrfeats=1000),
    2: WorkflowConfig(dropstop=True, synonyms=True, parents=False, ngrams=False, nrfeats=1000),
    3: WorkflowConfig(dropstop=True, synonyms=True, parents=True, ngrams=False, nrfeats=1000),
    4: WorkflowConfig(dropstop=True, synonyms=False, parents=False, ngrams=3, nrfeats=1000),
    5: WorkflowConfig(dropstop=True, synonyms=True, parents=False, ngrams=3, nrfeats=1000),
    6: WorkflowConfig(dropstop=True, synonyms=True, parents=True, ngrams=3, nrfeats=1000),
    7: WorkflowConfig(dropstop=False, synonyms=True, parents=True, ngrams=3, nrfeats=1000),
    8: WorkflowConfig(dropstop=False, synonyms=False, parents=False, ngrams=3, nrfeats=1000),
    9: WorkflowConfig(dropstop=False, synonyms=False, parents=False, ngrams=3, nrfeats=2000),
    10: WorkflowConfig(dropstop=True, synonyms=True, parents=True, ngrams=3, nrfeats=2000),
}


@dataclass
class ProjectPaths:
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    @property
    def abstracts_path(self) -> Path:
        return self.data_dir / "abstracts.tsv"

    @property
    def neo_path(self) -> Path:
        return self.data_dir / "neo.json"

    @property
    def med_stopwords_path(self) -> Path:
        return self.data_dir / "med-stopwords.txt"

    def ensure_dirs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


NUMPY_SEED = 41
TF_SEED = 42
SPLIT_SEED = 42
