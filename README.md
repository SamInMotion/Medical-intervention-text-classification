# Medical Intervention Text Classification

Automated screening of PubMed abstracts for systematic reviews on dementia and cognitive decline interventions. Classifies abstracts as relevant (include) or irrelevant (exclude) using logistic regression with ontology-based feature enrichment.

Developed as part of an MPhil thesis in Linguistics at the University of Bergen (2021-2024).

## What this project does

Given a set of PubMed abstracts, the pipeline tokenizes them, optionally enriches them with synonyms and parent concepts from the NEO ontology (a SNOMED-CT subset with 1,611 medical concepts), and classifies them using logistic regression. The ontology enrichment compensates for vocabulary mismatch in small datasets by adding semantic context that a bag-of-words model would otherwise miss.

The best-performing configuration (stopword removal + synonym enrichment + parent enrichment + trigrams + 2000 features + L2 regularization) reaches 90% accuracy on 150 manually annotated abstracts.

## Project history

The original thesis code lives in the Jupyter notebooks at the repo root. These are the notebooks I submitted with my thesis and they run the full experiment end-to-end.

In 2026 I started refactoring the notebook code into a proper Python package (`src/`) with tests, a CLI interface, and clearer separation between data loading, preprocessing, feature extraction, and evaluation. The goal is to make the pipeline easier to extend with k-fold cross-validation, benchmark datasets, and eventually transformer-based models.

**Original thesis code (2023):**
- `Main Classify Abstracts Code.ipynb` - full classification pipeline
- `Ontology Preferred Label Groupings.ipynb` - NEO ontology processing
- `classify_abstracts_new.py` - standalone script version

**v2.0 refactor (2026):**
- `src/` - modular package (config, data loading, preprocessing, features, models, evaluation, pipeline)
- `tests/` - pytest suite for preprocessing functions
- CLI entry point: `python -m src.pipeline --workflow 10`

## Setup

```bash
git clone https://github.com/SamInMotion/Medical-intervention-text-classification.git
cd Medical-intervention-text-classification
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place the data files (`abstracts.tsv`, `neo.json`, `med-stopwords.txt`) in the `data/` directory. See `data/README.md` for details.

## Usage

```bash
# run the best-performing workflow
python -m src.pipeline

# run a specific workflow configuration (0-10)
python -m src.pipeline --workflow 6

# run tests
pytest
```

## Workflow configurations

11 preprocessing configurations testing the impact of each enrichment strategy:

| ID | Stopwords | Synonyms | Parents | N-grams | Features |
|----|-----------|----------|---------|---------|----------|
| 0  | No        | No       | No      | No      | 1000     |
| 1  | Yes       | No       | No      | No      | 1000     |
| 2  | Yes       | Yes      | No      | No      | 1000     |
| 3  | Yes       | Yes      | Yes     | No      | 1000     |
| 4  | Yes       | No       | No      | 3       | 1000     |
| 5  | Yes       | Yes      | No      | 3       | 1000     |
| 6  | Yes       | Yes      | Yes     | 3       | 1000     |
| 7  | No        | Yes      | Yes     | 3       | 1000     |
| 8  | No        | No       | No      | 3       | 1000     |
| 9  | No        | No       | No      | 3       | 2000     |
| 10 | Yes       | Yes      | Yes     | 3       | 2000     |

## What I'm working on next

- k-fold cross-validation (the original evaluation used a single train/dev/test split, which the thesis examiners flagged)
- Benchmark dataset integration (Cohen et al. 2006, CLEF eHealth TAR 2017-2019) for comparison against established baselines
- BioBERT/PubMedBERT fine-tuning to compare transformer approaches against the ontology enrichment method
