# Data

## `cohen/epc-ir.clean.tsv`

The Cohen et al. (2006) drug-class benchmark file. Five tab-separated columns with
no header: topic, EndNote identifier, PubMed identifier, abstract-level triage
decision, article-level triage decision. Decisions are `I` for include and `E` for
exclude.

`src/benchmark_loader.py` maps `I` to 1 and everything else to 0. A minority of
rows carry numerals rather than `E` or `I` in the decision columns and are
therefore treated as exclusions; this is recorded as an open item in
`PROVENANCE.md`.

The three topics used in the manuscript are `Statins`, `Opiods` (the benchmark's
spelling) and `ADHD`.

## `cohen/pubmed_cache/`

6,216 PubMed records fetched through NCBI Entrez, one JSON file per PMID, each
holding the title, abstract, and expert-assigned MeSH terms.

**The pipeline does not read this directory.** It reads `cohen/cache/`, which is
excluded from version control. Copy the records across once before running
anything:

```bash
cp -r data/cohen/pubmed_cache data/cohen/cache
```

Without the copy, `src/benchmark_loader.py` re-fetches every record from Entrez on
first run. That works, but takes roughly an hour and requires an email address
passed via `--email` to comply with the Entrez usage policy.

The manuscript reports 2,744 Statins, 1,772 Opioids and 803 ADHD articles against
benchmark totals of 3,465, 1,915 and 851. The difference is records for which no
abstract was retrievable at the time of collection.
