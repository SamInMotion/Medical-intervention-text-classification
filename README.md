# Medical Intervention Text Classification

Automated screening of PubMed abstracts for systematic reviews on dementia and cognitive decline interventions. Classifies abstracts as relevant (include) or irrelevant (exclude) using logistic regression with ontology-based feature enrichment, with an extension to BiomedBERT on the Cohen et al. (2006) drug-class benchmark.

Developed as part of an MPhil thesis in Linguistics at the University of Bergen (2021-2024) and extended in 2026 with reproducibility infrastructure and a transformer-based comparison.

## What this project does

The project has two related strands.

The first is the original thesis pipeline: given a set of PubMed abstracts on dementia interventions, the pipeline tokenises them, optionally enriches them with synonyms and parent concepts from the NEO ontology (a SNOMED-CT subset with 1,611 medical concepts), and classifies them using regularised logistic regression. The ontology enrichment compensates for vocabulary mismatch in small datasets by adding semantic context that a bag-of-words model would otherwise miss. Five-fold stratified cross-validation across all 11 preprocessing configurations gives 80 to 86% accuracy on 150 manually annotated abstracts. The simplest configurations (raw trigrams, no ontology enrichment) perform as well as or better than enriched ones. This finding revised the original thesis results.

The second strand extends the methodology to the Cohen et al. (2006) drug-class benchmark and adds a transformer-based classifier comparison. The Cohen benchmark provides 15 drug-class systematic review topics with PubMed identifiers, inclusion labels, and expert MeSH annotations from the NLM indexers. The extension uses three of those topics (Statins, Opioids, ADHD) to ask a specific question: when the input text is augmented with MeSH terms, does it matter whether those terms come from human expert annotation or from mechanical substring matching against the same vocabulary? The bag-of-words pipeline shows a +0.096 WSS@95% advantage for expert MeSH on Statins, characterised across seven reruns of the same configuration. BiomedBERT under the same fold structure shows no detectable expert-vs-auto gap on any of the three topics, with a pooled 95% confidence interval that does not overlap the bag-of-words range. The paper draft is in internal review.

## Project history

The original thesis code lives in the Jupyter notebooks at the repo root. These are the notebooks submitted with the thesis and they run the full experiment end-to-end.

In 2026 the notebook code was refactored into a Python package (`src/`) with tests, a CLI interface, and clearer separation between data loading, preprocessing, feature extraction, and evaluation. The refactor enabled the Cohen benchmark extension and the BiomedBERT comparison.

**Original thesis code (2023):**
- `Main Classify Abstracts Code.ipynb` — full classification pipeline
- `Ontology Preferred Label Groupings.ipynb` — NEO ontology processing
- `classify_abstracts_new.py` — standalone script version

**v2.0 refactor (2026):**
- `src/pipeline.py` — original thesis pipeline, refactored
- `src/cohen_pipeline.py` — Cohen benchmark BoW pipeline
- `src/cohen_bert_pipeline.py` — Cohen benchmark BiomedBERT pipeline
- `src/benchmark_loader.py` — NCBI Entrez fetcher with local caching
- `src/auto_mesh.py` — substring-match MeSH assignment
- `src/features.py` — feature extraction (sklearn CountVectorizer replacement for original Keras Tokenizer)
- `src/bert_models.py` — BiomedBERT classifier wrapping HuggingFace transformers
- `src/evaluation.py` — WSS@95% and other screening metrics
- `tests/` — pytest suite for preprocessing functions

## Cohen benchmark extension (2026)

The extension addresses two open questions left by the thesis work: whether the ontology enrichment finding generalises beyond the dementia corpus, and whether the bag-of-words ceiling can be lifted with a contextual transformer.

The benchmark loader pulls article text and MeSH annotations from NCBI Entrez via Biopython, caches them locally, and matches them against the Cohen et al. (2006) inclusion labels. Three topics serve as the test set:

| Topic | Articles | Included | Inclusion rate |
|-------|----------|----------|----------------|
| Statins | 2,744 | 152 | 5.5% |
| Opioids | 1,772 | 43 | 2.4% |
| ADHD | 803 | 83 | 10.3% |

The bag-of-words pipeline runs Workflow 8 (the thesis-identified best configuration: trigrams, no stop-word removal, no ontology enrichment, 1000 features) on each topic with four text modes:

1. `abstract` — abstract text alone
2. `title_abstract` — title plus abstract
3. `title_abstract_mesh` — title plus abstract plus expert-assigned MeSH terms
4. `auto_mesh` — abstract plus mechanically-assigned MeSH terms via substring matching

The expert-vs-auto comparison is the central manipulation. The two modes draw from the same MeSH vocabulary with different assignment mechanisms.

The BiomedBERT pipeline uses `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` with standard fine-tuning hyperparameters (learning rate 2e-5, batch size 16, 3 epochs, max length 512, fp16). Five-fold stratified cross-validation, balanced class weights, and explicit `--seed 42` make the per-fold values reproducible from saved JSON sibling outputs.

**Headline result.** The bag-of-words classifier produces a Statins-specific expert-vs-auto MeSH gap of +0.096 WSS@95% on average across seven pipeline reruns (per-run range [+0.077, +0.114], all seven runs run-level positive, 32 of 35 fold-level paired differences strictly positive). BiomedBERT under the same fold structure shows no detectable expert-vs-auto gap on any of the three topics. The pooled across-three-topics BERT 95% confidence interval is [-0.098, +0.008], which does not overlap the bag-of-words multi-run range. The classifier absorbs the assignment-mechanism gap.

## Reproducibility infrastructure (2026)

The BiomedBERT pipeline writes per-fold WSS@95%, AUC, and accuracy values to `.json` sibling files alongside the standard `.txt` outputs. This enables downstream statistical analysis (bootstrap CIs, exact paired permutation tests, Nadeau-Bengio corrected t-tests) without re-running the GPU pipeline. The audit notebook (`notebooks/cohen_bert_audit.ipynb`) re-runs the three-topic single-seed analysis with explicit `--seed 42` and compares against pre-audit runs to document per-fold drift between implicit-RNG and explicit-seed conditions.

The bag-of-words pipeline exhibits residual non-determinism between identical-command reruns on the current Windows TensorFlow build, traced to Keras Dense layer initialisation in the BL Baseline component. The pipeline's existing `set_seeds()` function (which calls `np.random.seed()` and `tf.random.set_seed()`) does not fully control layer init. The methodological response is multi-run characterisation: seven reruns of the same configuration give a per-run mean and range for the central statistic, with the per-run variance reported alongside. The Statins multi-run analysis is complete. Multi-run characterisation of Opioids and ADHD is queued.

The full statistical analysis script (`demo_statistical_analysis.py`) parses the JSON sibling outputs and reproduces all tables in the paper draft. The multi-run parser (`parse_bow_multirun.py`) characterises BoW Statins reruns from `.txt` outputs.

## Setup

```bash
git clone https://github.com/SamInMotion/Medical-intervention-text-classification.git
cd Medical-intervention-text-classification
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place the original thesis data files (`abstracts.tsv`, `neo.json`, `med-stopwords.txt`) in the `data/` directory. See `data/README.md` for details.

The Cohen benchmark loader fetches and caches abstracts on first run. An NCBI Entrez email address is required (passed via `--email` on the command line) to comply with the Entrez usage policy.

## Usage

**Original thesis pipeline:**

```bash
# run the best-performing workflow on the dementia corpus
python -m src.pipeline

# run a specific workflow configuration (0-10)
python -m src.pipeline --workflow 6

# k-fold stratified cross-validation across all workflows with ROC AUC
python -m src.pipeline --kfold 5 --all-workflows
```

**Cohen benchmark BoW pipeline:**

```bash
# run all four text modes on the Statins topic
python -m src.cohen_pipeline \
  --topic Statins \
  --email your.email@example.com \
  --compare-text-modes \
  --output-file outputs/bow_statins.txt

# same for Opiods and ADHD (note the original Cohen TSV spelling of "Opiods")
python -m src.cohen_pipeline --topic Opiods --email your.email@example.com --compare-text-modes
python -m src.cohen_pipeline --topic ADHD --email your.email@example.com --compare-text-modes
```

**Cohen benchmark BiomedBERT pipeline (Colab T4 recommended):**

```bash
python -m src.cohen_bert_pipeline \
  --topic Statins \
  --email your.email@example.com \
  --text-mode title_abstract_mesh \
  --seed 42 \
  --output-file outputs/bert_statins_title_abstract_mesh_seed42.txt
```

The BiomedBERT pipeline writes per-fold raw values to a `.json` sibling next to the `.txt` output. The audit notebook (`notebooks/cohen_bert_audit.ipynb`) orchestrates the full three-topic by four-mode by single-seed audit run.

**Run tests:**

```bash
pytest
```

## Workflow configurations

Eleven preprocessing configurations testing the impact of each enrichment strategy on the dementia corpus.

| ID | Stopwords | Synonyms | Parents | N-grams | Features |
|----|-----------|----------|---------|---------|----------|
| 0 | No | No | No | No | 1000 |
| 1 | Yes | No | No | No | 1000 |
| 2 | Yes | Yes | No | No | 1000 |
| 3 | Yes | Yes | Yes | No | 1000 |
| 4 | Yes | No | No | 3 | 1000 |
| 5 | Yes | Yes | No | 3 | 1000 |
| 6 | Yes | Yes | Yes | 3 | 1000 |
| 7 | No | Yes | Yes | 3 | 1000 |
| 8 | No | No | No | 3 | 1000 |
| 9 | No | No | No | 3 | 2000 |
| 10 | Yes | Yes | Yes | 3 | 2000 |

The Cohen benchmark extension fixes Workflow 8 (the thesis-identified best) and varies the text mode and the classifier instead.

## What's next

- **Multi-run characterisation for BoW Opioids and ADHD.** Mirrors the Statins multi-run analysis so the cross-classifier comparison rests on characterised distributions at every topic. Approximately 30 minutes of compute per topic on the local machine. Methodologically appropriate before the paper goes external.

- **Multi-seed BiomedBERT for Opioids and ADHD.** The multi-seed Statins analysis exists (five seeds: 42, 7, 13, 21, 31). Extending it to the other two topics would tighten the null-finding CI. Approximately 5 hours T4 GPU. Not blocking the internal review.

- **Forest-plot figure for the paper.** BoW vs BERT expert-vs-auto gap by topic, with multi-run ribbon on Statins. Queued.

- **Extension to remaining Cohen topics.** Twelve more topics are available in the benchmark. Adding them would tighten the pooled CIs substantially and let the cross-classifier comparison rest on a wider topic base.

- **Alternative biomedical ontologies.** The thesis tested NEO. The Cohen extension tests MeSH. Whether the absorption finding holds with SNOMED-CT, UMLS, or domain-specific ontologies is an open question that requires either a new benchmark or a re-annotation of an existing one.

- **Alternative transformer architectures.** BiomedBERT-base is one model in a wider family. BioBERT, BiomedBERT-large, BioLinkBERT, and recent biomedical large language models would be natural comparators.

## Citation

If you use this work, please cite:

```
Okoe-Mensah, S. (2024). Medical intervention text classification using ontology-enriched
bag-of-words features. MPhil thesis, University of Bergen.
```

The Cohen benchmark extension paper is in internal review. Citation details will be added when the paper is publicly available.

## Acknowledgements

The original thesis work was supervised by Prof. Koenraad De Smedt (University of Bergen) and examined by Christer Johansson (University of Bergen). The 2026 extension benefited from internal review by Christer Johansson. The NEO ontology and the dementia abstracts collection were prepared by collaborators whose work the thesis built on.

## License

[Specify license — MIT, Apache 2.0, or other]
