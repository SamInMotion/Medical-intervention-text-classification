# Data

This directory should contain the following files (not included in the repository).

## Required files

**abstracts.tsv** - 150 manually annotated PubMed abstracts. Tab-separated, columns: `labels` (1=included, 0=excluded), `texts` (full abstract text). Source: curated for the MPhil thesis at University of Bergen.

**neo.json** - NEO (Neurological Examination Ontology) dictionary derived from SNOMED-CT. Maps preferred labels to synonyms, parent concepts, and sibling concepts. Contains 1,611 concepts.

**med-stopwords.txt** - 104 domain-specific medical stopwords, one per line.
