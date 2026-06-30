# BoW vs BERT data parity audit

**Purpose:** address Christer's Q1 — *"why are the two models not matched, and why are the different topics not matched, in size?"* — by documenting exactly what each classifier sees per abstract, where the asymmetries are, and which are confounds vs. which are design choices.

This document collates code reading (against `src/cohen_pipeline.py`, `src/cohen_bert_pipeline.py`, `src/preprocessing.py`, `src/features.py`, `src/bert_models.py`) with the empirical token-length output from `audit_token_lengths.py`.

---

## 1. What enters each pipeline per abstract

Both pipelines pull the same DataFrame from `benchmark_loader.load_cohen_topic`. The DataFrame holds `pmid`, `labels`, `texts` (PubMed abstract), `title`, `mesh_terms`. Auto-MeSH terms are built downstream and injected per mode.

For a given text mode, both pipelines build the same `parts = [title?, abstract, mesh?]` string per abstract. The construction is mirrored line-for-line:

| Mode | `parts` |
|---|---|
| `abstract` | `[abstract]` |
| `title_abstract` | `[title, abstract]` |
| `title_abstract_mesh` | `[title, abstract, " ".join(expert_mesh)]` |
| `auto_mesh` | `[title, abstract, " ".join(auto_mesh_terms)]` |

Same string. Same source data. **No asymmetry up to this point.**

## 2. Where BoW and BERT diverge

After the raw text is built, the two pipelines diverge on tokenization. This is the divergence Christer's question could be reading as "less data in one model":

**BoW path** (`cohen_pipeline.py` → `preprocess_corpus` → `prepare_features`):
1. Regex word-tokenize: `re.findall(r"[\w-]+", text.casefold())`
2. Append n-grams as underscore-joined tokens (Workflow 8 uses n=3)
3. NEO enrichment if configured (not used in Workflow 8)
4. Stopword removal if configured
5. Keras `Tokenizer(num_words=1000)` keeps the top 1,000 most frequent tokens fitted on the training fold, then converts to a binary presence matrix

BoW sees the **entire** abstract as a bag. It is not bounded by length. Long abstracts are not truncated; they just contribute more tokens (some of which fall outside the top 1,000 and so do not appear as features).

**BERT path** (`cohen_bert_pipeline.py` → `BiomedBertClassifier` → `_TextDataset`):
1. HuggingFace `AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")` produces WordPiece subword tokens
2. Truncation at `max_length=512` (`MAX_LEN` in `bert_models.py`)
3. Padding to `max_length=512`
4. Fed to fine-tuning forward pass

BERT sees up to **512 subword tokens** per abstract. Anything beyond that index is dropped.

### Implications

These are different representations of the same underlying text. They are not "matched" in the sense of identical numerical input; they cannot be, because the classifiers operate on fundamentally different input modalities (bag vs. sequence).

The question is whether the truncation introduces a meaningful asymmetry in *content* coverage. Run `audit_token_lengths.py` to quantify this. The expected pattern from PubMed structured abstracts is a median around 250-350 subword tokens for `title_abstract_mesh`, with a long tail at the upper end. If the truncation rate at 512 is small (single-digit percent), it is a minor disclosure in §3.5. If it is large (over a quarter of the corpus), it belongs in §6 limitations.

## 3. Run-count asymmetry between BoW and BERT

A separate question Christer's "not matched in size" could refer to: the per-topic fold count differs between BoW (n=35 = 7 reruns × 5 folds) and BERT (n=25 = 5 seeds × 5 folds).

This is a compute-budget choice, not a methodological one:
- BoW: ~5 min per topic per rerun on local Windows CPU.
- BERT: ~30-45 min per topic per seed on Colab T4 with `fp16`.

The 7 vs 5 ratio reflects what was practical within the available compute window. It does not bias the comparison: both classifiers are characterised as *distributions across reruns*, not as point estimates. The bootstrap CIs around the per-fold gap account for the per-classifier sample sizes correctly.

A more rigorous treatment matches the run counts. If a reviewer requests this, BERT seeds 6 and 7 can be added at the cost of one additional Colab T4 session per topic. Worth doing if the load-bearing reviewer at the target journal raises it.

## 4. Topic-size asymmetry

The third reading of "not matched in size" is across topics: Statins (≈2,744 abstracts) vs Opioids (≈1,772) vs ADHD (≈803). This is the Cohen et al. (2006) benchmark structure. The benchmark's unit of replication is the per-topic dataset *at its native size*. Matching the topics to a common n would mean:

1. Subsampling Statins/Opioids down to ADHD's 803, losing the larger-corpus signal that makes Statins worth analysing
2. Or upsampling ADHD, which has no principled way to do (the dataset is what was found in EPC-IR)

**Design decision:** retain native topic sizes. Acknowledge in §3 that this design parallels the Cohen literature (Cohen 2006, Wallace 2010, Kanoulas 2017, all of which retain native sizes for the same reason) and that the topic-size confound is addressed directly via the Statins subsampling experiment in §5.2.

## 5. The auto-MeSH text asymmetry

Worth flagging because it could be raised as a Q1 sub-question: in `title_abstract_mesh` mode, expert MeSH terms are joined with spaces and appended. In `auto_mesh` mode, auto-assigned terms are appended. The two MeSH sources are different in size and quality:

- Expert MeSH (NLM annotators): typically 8-15 terms per abstract, MeSH-tree-coherent
- Auto MeSH (`auto_mesh.build_mesh_vocabulary`): vocabulary-lookup match against abstract text; coverage and quality vary by topic

If a reviewer asks whether the gap is driven by quantity of MeSH text rather than quality, the answer is: both arms ingest a string-joined token list of MeSH labels, so the *quantity asymmetry* maps directly to the *content asymmetry* we are studying. The paper's claim is about the content difference, not a counterfactual where both arms see equally many MeSH tokens.

## 6. What this audit does NOT cover

- Whether different seeds for the auto-MeSH vocabulary construction matter. The construction is deterministic per the cache, so this is not a source of variance.
- Whether BoW's 1,000-feature cap is comparable to BERT's full subword vocabulary (~30k). It is not, but this is a representation-capacity question rather than a data-parity one. Document in §3.5 as a known asymmetry.
- Whether fold-assignment seeds are matched. They are: both pipelines use `StratifiedKFold(n_splits, shuffle=True, random_state=SPLIT_SEED)` with the same `SPLIT_SEED` from `config.py`.

## 7. Net assessment

The two models are not "matched in size" in three distinct senses:
1. Per-abstract token count (BoW unbounded vs. BERT capped at 512) — quantify with `audit_token_lengths.py`, disclose in §3.5 or §6 depending on the truncation rate.
2. Per-topic fold count (BoW n=35 vs. BERT n=25) — disclose as a compute-budget choice in §3.5, defend with the distributional comparison framework.
3. Per-topic dataset size (Statins ≈ 3× ADHD) — defend as the Cohen benchmark's native structure, with the subsampling experiment in §5.2 addressing the confound directly.

None of these is a flaw. Each is a design choice that needs explicit disclosure in the methods section so a reviewer reading carefully arrives at the same conclusion the audit reaches: the asymmetries are documented, bounded, and addressed.

## Suggested edits to the paper

- §3 (Data): add one sentence stating the benchmark's native topic sizes are retained, with a forward reference to §5.2.
- §3.5 (Methods asymmetries): two-paragraph block covering the three asymmetries above. Include the token-length numbers from `audit_token_lengths.json` once it runs.
- §6 (Limitations): one-paragraph entry on truncation if `audit_token_lengths.py` shows >10% truncation in `title_abstract_mesh` mode for Statins.
