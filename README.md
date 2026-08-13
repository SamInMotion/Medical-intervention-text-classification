# MeSH assignment and evaluation design in systematic review screening

Code, data, and per-fold outputs for a manuscript under anonymous review.

The manuscript asks whether expert-assigned and mechanically-assigned MeSH terms
are interchangeable as classifier features for systematic review screening, and
whether the answer depends on how the classifier is evaluated. Experiments run on
three drug-class topics from the Cohen et al. (2006) benchmark, comparing a
bag-of-words logistic regression classifier against BiomedBERT across four input
representations.

Everything needed to reproduce every table and figure in the manuscript is here.

---

## Before you run anything

The benchmark cache is distributed at `data/cohen/pubmed_cache/`, but the pipeline
reads `data/cohen/cache/`. Copy it once:

```bash
cp -r data/cohen/pubmed_cache data/cohen/cache
```

Without this the loader re-fetches 6,216 records from NCBI Entrez, which takes
roughly an hour and requires an email address for their usage policy.

```bash
git clone https://anonymous.4open.science/r/mesh-assignment-screening.git
cd mesh-assignment-screening
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp -r data/cohen/pubmed_cache data/cohen/cache
```

---

## Layout

```
PROVENANCE.md            every number in the manuscript, mapped to its source file
REPRODUCING.md           claim-to-command map, generated from PROVENANCE.md
src/                     pipelines, benchmark loader, MeSH assignment, metrics
tests/                   pytest suite
data/cohen/              benchmark TSV and cached PubMed records
outputs/                 per-fold results, summaries, figures
outputs/archive/         superseded runs, retained as experimental record
paper_experiments/       robustness analyses, power analysis, token-length audit
notebooks/               Colab notebooks for the GPU runs
scripts/                 figure generators and the multi-run driver
```

`PROVENANCE.md` is authoritative. It carries one row per number in the manuscript
with its source file, that file's date, the run conditions, and a verification
status, along with the defects still open against the code and documentation.
`REPRODUCING.md` is generated from it; where the two disagree, `PROVENANCE.md`
wins.

---

## Running the experiments

Bag-of-words, all four text modes on one topic:

```bash
python -m src.cohen_pipeline \
  --topic Statins \
  --email your.address@example.com \
  --compare-text-modes \
  --output-file outputs/bow_statins.txt
```

Topic names follow the benchmark file, which spells Opioids as `Opiods`.

Seven-rerun characterisation, used for the multi-run tables:

```bash
EMAIL=your.address@example.com bash scripts/run_bow_multirun.sh
python parse_bow_multirun.py outputs/bow_statins_run*.txt
```

BiomedBERT requires a GPU. Both notebooks under `notebooks/` were executed on
Colab T4; the multi-seed run takes roughly five hours for fifty fine-tunings.
Per-fold values from those runs are archived under `outputs/`, so the statistical
analyses can be reproduced without repeating the GPU work.

MeSH assignment analysis:

```bash
python -m paper_experiments.mesh_assignment_analysis --email your.address@example.com
```

Figures:

```bash
python scripts/make_fig1_gap_forest.py       # reads the six summary JSONs
python make_fig2_design_sensitivity.py       # values hardcoded, see PROVENANCE.md
```

---

## Notes

Some modules under `src/` belong to an earlier project from which this pipeline
derives and are not used by any analysis reported in the manuscript:
`preprocessing.py` and `models.py`. They are
retained because `src/cohen_pipeline.py` imports feature-extraction helpers from
the same package.

`scripts/make_fig1_v2.py` and `scripts/make_paper_artifacts.py` predate the
current output naming and are retained as record, not as regeneration paths.

The bag-of-words pipeline is not fully deterministic across identical-command
reruns; the cause and the multi-run response are documented in the manuscript and
in `PROVENANCE.md`.

---

## License

MIT. See `LICENSE`.
