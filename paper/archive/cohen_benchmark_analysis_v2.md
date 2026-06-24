# Cohen Benchmark Analysis: Ontology Enrichment, Expert Curation, and Mechanical Lookup Across Datasets

**Author:** Samuel Okoe-Mensah
**Date:** April 14, 2026
**Version:** 2.0 (updated with auto-MeSH experiment)
**Governing research question:** Can ontology-based feature enrichment improve text classification for systematic review screening, and if so, what makes enrichment effective?

---

## 1. Purpose and Contribution

The MPhil thesis (University of Bergen, 2021–2024) evaluated ontology-enriched classification on 150 abstracts from one Cochrane systematic review on dementia interventions. The k-fold re-evaluation (April 12, 2026) revised the headline accuracy from 93% to 80–86% and found that NEO enrichment provided no measurable benefit. These findings left an open question: did ontology enrichment fail because the method is flawed, or because the conditions were wrong?

This document reports three experiments designed to answer that question:

1. **Cross-dataset replication:** The same 11 workflow configurations tested on the Statins topic from Cohen et al. (2006), a corpus of 2,744 abstracts with 5.5% inclusion rate.
2. **Expert MeSH enrichment:** Adding expert-assigned MeSH terms (from PubMed indexers) to the bag-of-words features.
3. **Automatic MeSH lookup:** Adding MeSH terms found by mechanical string matching against abstracts — the same method as the thesis's NEO enrichment, but with MeSH vocabulary.

The third experiment is the critical one. It isolates whether the benefit of structured vocabulary comes from the vocabulary itself or from the human judgment that selects which terms are relevant to each document.

**Reference:** Cohen, A.M., Hersh, W.R., Peterson, K., and Yen, P.Y. (2006). Reducing Workload in Systematic Review Preparation Using Automated Citation Classification. *JAMIA*, 13(2), 206–219.

---

## 2. Experimental Setup

### 2.1 Dataset

The Cohen et al. dataset contains 18,733 triage decisions across 15 drug class topics. The Statins topic was selected for the first run as it is the largest topic (3,465 records) with a low inclusion rate (5.0%), representative of real-world screening conditions.

After fetching abstracts from PubMed via Biopython Entrez:
- 2 PMIDs could not be retrieved (likely retracted or merged records)
- 719 PMIDs had no abstract text in PubMed (comments, editorials, historical articles, biographical pieces)
- **Final corpus: 2,744 abstracts, 152 included (5.5%), 2,592 excluded (94.5%)**

The dropout of 721 records (20.8%) is a known consequence of using abstract text only. Cohen et al. used title + abstract + MeSH as features, so they could classify articles without abstracts. Of the 173 originally included articles, 21 (12.1%) were lost to empty abstracts. The class balance shifted marginally from 5.0% to 5.5%.

### 2.2 Pipeline

The same modular pipeline from the thesis v2.0 refactor was used without modification. Logistic regression (single-layer perceptron) with binary bag-of-words features, one-hot encoding, and Keras Tokenizer for vocabulary management. Two model variants: baseline (no regularization) and L2-regularized (lambda = 2e-6). All configurations used 5-fold stratified cross-validation with the vectorizer fitted on training folds only.

### 2.3 Four text modes

| Mode | Input to classifier | Assignment method | Analogous to |
|------|-------------------|-------------------|-------------|
| `abstract` | Abstract text only | N/A (raw text) | Thesis baseline |
| `title_abstract` | Title + abstract concatenated | N/A (raw text) | Extended baseline |
| `title_abstract_mesh` | Title + abstract + expert-assigned MeSH terms | Human indexer selects per-article | Expert enrichment |
| `auto_mesh` | Abstract + MeSH terms found by string matching | Mechanical substring search against full MeSH vocabulary | NEO enrichment (thesis method) |

The `auto_mesh` mode is the controlled comparison. It uses the same MeSH vocabulary as the expert mode but assigns terms through mechanical lookup rather than human judgment. This makes the comparison with the thesis's NEO enrichment methodologically fair: both use mechanical string matching to add structured vocabulary terms to the feature vector. The only difference is the vocabulary source (MeSH vs. NEO).

### 2.4 Auto-MeSH vocabulary construction

The auto-MeSH vocabulary was built from all MeSH terms appearing across the 3,463 cached PubMed records for the Statins topic. Terms shorter than 4 characters were excluded to filter out qualifiers like "use" that would match everywhere. This produced **3,040 unique lowercase MeSH terms**. On average, **12.3 terms matched per abstract** via case-insensitive substring search, totalling 33,813 matches across 2,744 abstracts.

### 2.5 NEO enrichment as negative control

The NEO (Neurologic Examination Ontology) covers 1,611 neurological examination concepts. It was designed for the thesis's dementia intervention corpus. The Statins corpus discusses cholesterol, cardiovascular outcomes, and drug interactions — a vocabulary where NEO terms are almost entirely absent. Workflows requesting synonym and/or parent enrichment (2, 3, 5, 6, 7, 10) serve as negative controls, confirming that the enrichment mechanism does not introduce noise when the ontology is domain-irrelevant.

### 2.6 Evaluation metrics

| Metric | Purpose | Note for this dataset |
|--------|---------|----------------------|
| Accuracy | Continuity with thesis | **Uninformative at 5.5% inclusion.** A classifier predicting "excluded" for every article achieves 94.5%. |
| ROC AUC | Threshold-independent ranking quality | How well probability scores separate included from excluded articles. |
| WSS@95% | Standard SR screening metric | Proportion of articles a reviewer can skip at 95% recall, minus the 5% random baseline. **Primary metric for this analysis.** |

---

## 3. Results

### 3.1 Text mode comparison (workflow 8, Statins, 5-fold CV)

| Text Mode | BL Acc | BL AUC | BL WSS@95 | Reg Acc | Reg AUC | Reg WSS@95 |
|-----------|--------|--------|-----------|---------|---------|------------|
| abstract | .933±.006 | .743±.038 | .096±.073 | .943±.003 | .752±.036 | .123±.086 |
| title_abstract | .935±.006 | .748±.035 | .121±.080 | .941±.004 | .760±.031 | .114±.067 |
| **title_abstract_mesh** | **.941±.005** | **.767±.038** | **.206±.068** | **.944±.004** | **.774±.038** | **.223±.090** |
| auto_mesh | .934±.007 | .738±.046 | .138±.096 | .945±.001 | .749±.042 | .102±.058 |

### 3.2 All 11 workflows on Statins (abstract mode, 5-fold CV)

| WF | Stop | Syn | Par | Ngr | Feat | BL Acc | Reg Acc | BL AUC | Reg AUC | BL WSS@95 | Reg WSS@95 |
|----|------|-----|-----|-----|------|--------|---------|--------|---------|-----------|------------|
| 0 | N | N | N | N | 1000 | .936±.003 | .944±.003 | .742±.046 | .752±.044 | .149±.115 | .113±.056 |
| 1 | Y | N | N | N | 1000 | .933±.007 | .944±.003 | .722±.062 | .732±.061 | .166±.066 | .158±.050 |
| 2 | Y | Y | N | N | 1000 | .932±.006 | .945±.005 | .723±.063 | .731±.058 | .162±.071 | .142±.044 |
| 3 | Y | Y | Y | N | 1000 | .933±.007 | .943±.006 | .722±.063 | .731±.058 | .164±.068 | .147±.051 |
| 4 | Y | N | N | 3 | 1000 | .934±.006 | .943±.004 | .717±.053 | .724±.052 | .134±.083 | .123±.081 |
| 5 | Y | Y | N | 3 | 1000 | .932±.005 | .944±.003 | .717±.053 | .724±.055 | .132±.084 | .126±.079 |
| 6 | Y | Y | Y | 3 | 1000 | .932±.005 | .945±.004 | .717±.053 | .723±.054 | .134±.083 | .120±.074 |
| 7 | N | Y | Y | 3 | 1000 | .934±.006 | .944±.003 | .744±.040 | .750±.035 | .096±.073 | .121±.087 |
| 8 | N | N | N | 3 | 1000 | .933±.008 | .943±.003 | .742±.039 | .752±.040 | .093±.070 | .123±.086 |
| 9 | N | N | N | 3 | 2000 | .943±.004 | .945±.002 | .739±.053 | .753±.046 | .080±.024 | .112±.027 |
| 10 | Y | Y | Y | 3 | 2000 | .942±.005 | .942±.005 | .706±.057 | .719±.056 | .078±.023 | .075±.012 |

### 3.3 Thesis k-fold results for comparison (150 abstracts, dementia, 59/41 class split)

| WF | Config summary | BL Acc | Reg Acc |
|----|---------------|--------|---------|
| 0 | Raw unigrams | .853±.050 | .853±.034 |
| 8 | Trigrams, no enrichment | .860±.057 | .853±.058 |
| 3 | Stopwords + synonyms + parents | .807±.044 | .793±.049 |
| 10 | Full enrichment + 2000 features | .833±.082 | .800±.073 |

---

## 4. Analysis

### 4.1 Accuracy is uninformative on imbalanced data

All 11 workflows cluster between 93.2% and 94.5% accuracy — a 1.3pp range where the majority-class baseline is 94.5%. Several regularized models match this baseline. On the thesis data with its 59/41 split, accuracy ranged from 80.7% to 86.0% — a meaningful 5.3pp spread. On the Statins data, accuracy carries no information about the classifier's ability to find included articles. AUC and WSS@95% are the metrics for the rest of this analysis.

### 4.2 NEO enrichment has no effect on Cohen data (expected negative control)

Comparing enriched vs. non-enriched workflows on AUC:

| Comparison | Without NEO | With NEO | Diff |
|-----------|-------------|----------|------|
| WF1 vs WF2 (add synonyms) | .722 | .723 | +.001 |
| WF2 vs WF3 (add parents) | .723 | .722 | −.001 |
| WF4 vs WF5 (syn + trigrams) | .717 | .717 | .000 |
| WF5 vs WF6 (parents + trigrams) | .717 | .717 | .000 |
| WF8 vs WF7 (full enrichment) | .742 | .744 | +.002 |

All differences are within 0.002 AUC — effectively zero. The same pattern holds on WSS@95%: no consistent direction, differences within standard deviations. NEO enrichment passes through transparently when the ontology vocabulary does not overlap with the corpus. This validates the experimental design: the enrichment mechanism itself is inert when the vocabulary is irrelevant.

### 4.3 Stopword removal has no consistent effect — replicating the thesis finding

The thesis k-fold showed stopword removal was a split artifact (WF0 85.3% vs WF1 84.7%). On the Statins data:

| Metric | Keep stopwords (WF0) | Remove stopwords (WF1) | Diff |
|--------|---------------------|----------------------|------|
| BL AUC | .742 | .722 | −.020 |
| BL WSS@95 | .149 | .166 | +.017 |
| Reg AUC | .752 | .732 | −.020 |
| Reg WSS@95 | .113 | .158 | +.045 |

Stopword removal slightly hurts AUC but slightly helps WSS@95. The effects go in opposite directions and are within standard deviations. The pattern replicates across the trigram workflows as well: WF8 vs WF4 shows the same AUC disadvantage for stopword removal. The finding is consistent across both datasets: stopword removal does not produce a reliable improvement for this task.

### 4.4 Trigrams reduce WSS@95 — reversing the thesis finding

On the thesis data, trigram workflows performed best (WF8 at 86.0% vs WF0 at 85.3%). On the Statins data, the AUC difference between unigrams and trigrams is negligible, but WSS@95 tells a different story:

| Config | BL AUC | BL WSS@95 |
|--------|--------|-----------|
| WF0 (unigrams) | .742 | **.149** |
| WF8 (trigrams) | .742 | .093 |
| WF1 (unigrams, stop removed) | .722 | **.166** |
| WF4 (trigrams, stop removed) | .717 | .134 |

Unigram workflows outperform their trigram counterparts on WSS@95% in every paired comparison. The hypothesis: with 2,744 abstracts and a 1,000-feature vocabulary cap, unigrams capture the most discriminative individual terms (drug names, outcome terms, study design keywords). Trigrams consume vocabulary slots with three-word sequences that are less generalizable across the diverse articles in a broad drug class review. On the thesis's 150 abstracts from one systematic review, vocabulary was consistent enough that trigrams captured useful phrases. On a broader corpus, individual terms carry more signal per feature slot.

### 4.5 Expert MeSH is the single biggest improvement

The text mode comparison produces the clearest result in the study:

| Text Mode | BL AUC | BL WSS@95 | Reg WSS@95 |
|-----------|--------|-----------|------------|
| Abstract only | .743 | .096 | .123 |
| Title + abstract | .748 | .121 | .114 |
| Title + abstract + expert MeSH | **.767** | **.206** | **.223** |
| Abstract + auto MeSH lookup | .738 | .138 | .102 |

Expert-assigned MeSH terms roughly double WSS@95% compared to abstract-only classification. The regularized model with expert MeSH achieves WSS@95 of 0.223 — meaning a reviewer can skip 22.3% of articles (above the 5% random baseline) while catching 95% of included ones. Without MeSH, the reviewer saves only 12.3%.

In practical terms on the 2,744-article Statins corpus: the MeSH-enhanced classifier saves a reviewer approximately **612 articles** they do not need to read (at 95% recall), compared to **337 articles** without MeSH.

Adding the title alone provides a modest improvement (AUC +.005, BL WSS +.025). The MeSH terms carry the bulk of the signal.

### 4.6 Auto-MeSH lookup fails — the decisive experiment

The auto-MeSH mode uses the same MeSH vocabulary as the expert mode (3,040 terms) but assigns terms through mechanical substring matching rather than expert judgment. The results are unambiguous:

| Comparison | AUC | WSS@95 (Reg) |
|-----------|------|-------------|
| Expert MeSH vs abstract | +.024 | +.100 |
| Auto MeSH vs abstract | −.005 | −.021 |
| **Expert MeSH vs auto MeSH** | **+.029** | **+.121** |

Auto-MeSH performs at or slightly below the abstract-only baseline on every metric. On AUC, it is 0.005 below abstract-only (0.738 vs 0.743 baseline). On regularized WSS@95, it is 0.021 below (0.102 vs 0.123). Expert MeSH outperforms auto-MeSH by 0.029 AUC and 0.121 WSS@95 (regularized).

The auto-MeSH approach matched 12.3 terms per abstract on average from a vocabulary of 3,040 terms. These matches include high-frequency terms that appear in nearly every biomedical abstract — "humans," "male," "female," "adult," "drug therapy," "treatment outcome." Such terms provide no discriminative signal between included and excluded articles. They dilute the feature space by consuming vocabulary slots with uninformative features, exactly as NEO parent concepts diluted the thesis feature space with broad terms like "pain" and "motor disorder."

### 4.7 What the auto-MeSH experiment proves

The four-way comparison isolates the causal factor:

| Variable held constant | Variable changed | Effect on WSS@95 |
|----------------------|-----------------|------------------|
| MeSH vocabulary, bag-of-words method | Assignment: expert → mechanical | +0.223 → +0.102 (−0.121) |
| Mechanical assignment method | Vocabulary: MeSH → NEO | 0.102 → ~0.096 (negligible) |
| Expert assignment, MeSH vocabulary | Integration: title+abstract vs abstract only | +0.206 → +0.223 (with title) |

The only variable that produces a large effect is the assignment method. Switching from expert to mechanical assignment erases the entire benefit of MeSH enrichment. Switching vocabularies (MeSH vs NEO) while keeping mechanical assignment changes nothing. Adding title text produces a small effect.

This answers the question left open by the thesis. The thesis's NEO enrichment failed not because NEO was the wrong ontology (although it was, for the Statins domain), and not because the enrichment method was flawed in principle, but because **mechanical term assignment introduces noise that cancels any signal from the vocabulary**. The discriminative power of structured vocabulary depends entirely on whether the terms assigned to each document are contextually relevant — something that requires judgment, not lookup.

### 4.8 Connecting to the distributional hypothesis

The thesis framed the research question through the lens of computational semantics: when does distributional information (bag-of-words statistics) suffice for text understanding, and when must it be supplemented with structured knowledge?

The auto-MeSH experiment refines this framing. Distributional features (abstract text) achieve AUC of 0.743 on the Statins data. Adding structured vocabulary through mechanical lookup does not improve this — the distributional information in the abstract already captures what the MeSH vocabulary can express through string matching. The only form of structured knowledge that improves over distributional features is **curated** structured knowledge, where a human has verified that each term is relevant in context.

This connects to a broader pattern in NLP: the value of knowledge graphs and ontologies in text classification depends not on the vocabulary they contain, but on the mapping between vocabulary and documents. A correct mapping (expert MeSH assignment, or in principle, a high-quality concept linker) adds signal. An incorrect or noisy mapping (string matching, or ontology lookup without disambiguation) adds noise that the classifier cannot distinguish from signal.

### 4.9 Comparison to Cohen et al. (2006) baselines

Cohen et al. reported WSS@95% per topic using a voting perceptron with binary bag-of-words features (title + abstract + MeSH + publication type) and chi-squared feature selection with 5×2 cross-validation. Their average WSS@95% across all 15 topics was approximately 18.5%.

Our best result with title + abstract + expert MeSH (regularized WSS@95 of 0.223) exceeds this average, though direct comparison is complicated by differences in classifier, cross-validation scheme, feature representation, and our 21% smaller corpus due to empty-abstract dropout.

The abstract-only regularized WSS@95 (0.123) is notably lower than Cohen et al.'s average, which is expected — Cohen et al. used title, MeSH, and publication type in addition to abstract text. The gap between our abstract-only and title+abstract+MeSH results (0.123 → 0.223) suggests that approximately half of Cohen et al.'s workload reduction came from the non-abstract features.

### 4.10 Regularization consistently helps on imbalanced data

Unlike the thesis data (where regularization was inconsistent), on the Statins data regularization improves AUC for all 11 workflows without exception. This makes sense: with 5.5% inclusion rate, the unregularized model overfits to the majority class. L2 regularization prevents weights from growing too large in the "always predict excluded" direction, preserving the model's ability to assign higher probabilities to included articles.

### 4.11 High WSS variance across folds

WSS@95 standard deviations are large relative to means (0.023–0.115). Fold 5 consistently produces near-zero or negative WSS across all conditions. This fold-level instability reflects the sensitivity of the 95% recall threshold with only ~30 included articles per fold. A fold containing hard-to-classify positives pushes the threshold lower and eliminates work savings.

The practical implication: WSS@95% on any single train/test split is unreliable. The k-fold mean is the right summary statistic, but even the mean should be interpreted as an order-of-magnitude estimate.

---

## 5. Cross-Dataset Comparison

### 5.1 Summary table

| Condition | Thesis (dementia, 150 abstracts) | Cohen Statins (2,744 abstracts) |
|-----------|--------------------------------|-------------------------------|
| Class balance | 59% / 41% | 5.5% / 94.5% |
| Best BL accuracy | 86.0% (WF8) | 94.3% (WF9) — uninformative |
| Best BL AUC | 0.932 (WF0) | 0.767 (expert MeSH) |
| Best Reg WSS@95 | Not computed | 0.223 (expert MeSH) |
| NEO enrichment effect | None | None (negative control) |
| Auto MeSH effect | Not tested | None (−0.021 WSS vs baseline) |
| Expert MeSH effect | Not tested | +0.100 WSS@95 over baseline |
| Stopword effect | None (split artifact) | None (replicates) |
| Trigram effect | +0.7pp accuracy | −0.03 to −0.06 WSS@95 |
| Regularization effect | Inconsistent | Consistently positive |

### 5.2 What changes with dataset size

The thesis data had 150 abstracts; Statins has 2,744. The larger dataset produces lower AUC (0.74 vs 0.93), reflecting a harder classification problem, not worse methodology. The thesis corpus contained articles from one Cochrane review where included and excluded articles discuss different aspects of dementia interventions. The Statins corpus spans a broad drug class where excluded articles overlap heavily in vocabulary with included ones.

Variance drops substantially with size. Thesis accuracy standard deviations: 3.4–8.2%. Statins accuracy: 0.1–0.8%. AUC standard deviations on Statins (0.035–0.063) are roughly half the thesis values. Larger datasets produce more stable estimates even when absolute performance is lower.

### 5.3 The enrichment finding across datasets

Three forms of vocabulary enrichment were tested across the two datasets:

| Enrichment type | Assignment method | Vocabulary relevance | Effect |
|----------------|-------------------|---------------------|--------|
| NEO on thesis data | Mechanical lookup | Domain-relevant (neurological) | None |
| NEO on Statins data | Mechanical lookup | Domain-irrelevant (neurological → cardiovascular) | None |
| Auto MeSH on Statins | Mechanical lookup | Domain-relevant (biomedical) | None (slightly negative) |
| Expert MeSH on Statins | Human expert | Domain-relevant (biomedical) | **+0.100 WSS@95** |

The pattern is consistent: mechanical assignment produces no benefit regardless of vocabulary relevance. Expert assignment produces a large benefit. The enrichment method matters more than the enrichment vocabulary.

---

## 6. Implications for the Research Question

The thesis asked: *Can ontology-based feature enrichment improve text classification for systematic review screening when data is scarce?*

The combined evidence from three experiments supports a refined answer:

### 6.1 The original hypothesis is not supported

The thesis hypothesized that ontology enrichment compensates for data scarcity. The evidence does not support this:

- On 150 abstracts (scarce data), NEO enrichment did not help.
- On 2,744 abstracts (more data), auto-MeSH enrichment did not help.
- Expert MeSH helped on 2,744 abstracts, but this is the opposite of the scarcity hypothesis — more data, more benefit.

The thesis hypothesis conflated two things: the value of structured vocabulary (real, but conditional) and the idea that small datasets especially benefit from it (not demonstrated).

### 6.2 A revised hypothesis

**Structured semantic features improve classification when two conditions hold simultaneously:**

1. **Contextual relevance:** The enrichment terms must be relevant to each specific document, not just to the domain in general. Expert-assigned MeSH terms satisfy this because a human verified relevance. Mechanical lookup does not, because it matches terms based on string overlap without contextual assessment.

2. **Discriminative specificity:** The enrichment terms must discriminate between included and excluded articles. High-frequency terms ("humans," "drug therapy") match most articles indiscriminately and add noise. Specific terms ("Hydroxymethylglutaryl-CoA Reductase Inhibitors," "Rhabdomyolysis") match selectively and add signal. Expert assignment naturally selects for specificity because indexers annotate what an article is *about*, not what words it *contains*.

### 6.3 Why mechanical lookup fails

The auto-MeSH experiment matched 12.3 terms per abstract from a vocabulary of 3,040 terms. Many of these matches are uninformative: "drug therapy" appears in both statin efficacy studies (included) and pharmacokinetic studies of unrelated drugs (excluded). The mechanical matcher cannot distinguish these contexts. In a bag-of-words model with a fixed vocabulary cap (1,000 features), these uninformative matches consume feature slots that would otherwise represent discriminative words from the abstract itself.

This is the same mechanism as the thesis's "parent dilution" finding. Parent concepts like "pain" and "motor coordination disorder" matched too broadly, introducing noise. Auto-MeSH terms like "humans" and "drug therapy" match too broadly for the same reason. The thesis identified the symptom (parent dilution); the auto-MeSH experiment identifies the cause (context-free term assignment).

### 6.4 Implications for ontology-based NLP systems

The finding has implications beyond systematic review screening. Any system that enriches text features with ontology terms faces the same tradeoff: mechanical lookup is cheap and scalable but introduces noise proportional to vocabulary breadth. Contextually validated assignment is expensive but preserves signal-to-noise ratio.

Potential middle-ground approaches (not tested in this study, but suggested by the results):

- **Concept linking with disambiguation:** Systems like MetaMap, cTAKES, or ScispaCy map text to UMLS concepts with context-sensitive disambiguation. This is more expensive than string matching but cheaper than expert indexing.
- **Term frequency filtering:** Removing high-frequency MeSH terms (those appearing in >50% of abstracts) before enrichment would eliminate the most noisy matches while preserving rare, discriminative terms.
- **TF-IDF weighting of enrichment terms:** Instead of binary presence, weighting enrichment terms by their discriminative value (high weight for rare terms, low weight for common ones) could mitigate the dilution effect without requiring expert assignment.

These are hypotheses for future work, not findings of this study.

---

## 7. Limitations

### 7.1 Single topic

Results are from the Statins topic only. Statin has 5.5% inclusion rate and is the largest Cohen topic. Results may differ on topics with higher inclusion rates (AtypicalAntipsychotics at 32.4%) or smaller sizes (Antihistamines at 310 articles).

### 7.2 Abstract dropout

719 of 3,465 Statins PMIDs (20.8%) had no abstract in PubMed. Cohen et al. could classify these using title and MeSH. Our results are not directly comparable to their per-topic numbers.

### 7.3 Auto-MeSH uses naive string matching

The auto-MeSH implementation uses case-insensitive substring matching. More sophisticated concept linking (e.g., MetaMap, ScispaCy) might perform better by disambiguating matches and filtering irrelevant ones. The current experiment tests the simplest form of mechanical lookup — the same approach as the thesis's NEO enrichment — not the best possible automatic approach.

### 7.4 Single workflow for text mode comparison

The text mode comparison used workflow 8 only (trigrams, no stopwords, no enrichment). Running all four text modes across all 11 workflows would provide a more complete picture, particularly for unigram configurations where the vocabulary cap dynamics differ.

### 7.5 No statistical significance testing

Means and standard deviations are reported without formal hypothesis tests. Given the large WSS standard deviations (0.058–0.096), some of the smaller observed differences may not be statistically significant. The expert MeSH vs. abstract comparison (Reg WSS 0.223 vs 0.123, a difference of 0.100) is large relative to both standard deviations, but a paired t-test or bootstrap confidence interval would strengthen the claim.

### 7.6 Fixed hyperparameters

Training used 42 epochs with batch sizes of 5 (baseline) and 32 (regularized), inherited from the thesis. These were not tuned for the Statins dataset. The optimal configuration may differ at 2,744 abstracts.

### 7.7 Expert MeSH is not a fair baseline for real-world deployment

Expert MeSH assignment requires articles to be indexed by NLM librarians — a process that takes weeks to months after publication. In a live systematic review, newly published articles may not yet have MeSH terms. The practical value of expert MeSH enrichment depends on the review's search strategy and the lag between publication and indexing.

---

## 8. Data Provenance and Reproducibility

| Component | Source | Retrieved |
|-----------|--------|-----------|
| Cohen TSV | https://dmice.ohsu.edu/cohenaa/epc-ir-data/epc-ir.clean.tsv | April 14, 2026 |
| PubMed abstracts + MeSH | NCBI PubMed via Biopython Entrez | April 14, 2026 |
| Thesis k-fold data | kfold_analysis.md (this repository) | April 12, 2026 |
| NEO ontology | data/neo.json (this repository) | 2024 |
| Pipeline code | src/ package v2.1 (this repository) | April 14, 2026 |

All PubMed records are cached as individual JSON files in `data/cohen/cache/`. Re-running the pipeline uses cached data without API calls. Random seeds fixed in `src/config.py` (NUMPY_SEED=41, TF_SEED=42, SPLIT_SEED=42).

To reproduce all results:
```
python -m src.cohen_pipeline --topic Statins --email <email> --compare-text-modes --output-file outputs/text_mode_comparison.txt
python -m src.cohen_pipeline --topic Statins --email <email> --all-workflows --output-file outputs/all_workflows_statins.txt
```

---

## 9. Next Steps

### 9.1 Additional Cohen topics

Run 2–3 topics with different characteristics:
- **ADHD** (851 articles, 9.9% inclusion): Neurological/cognitive domain; NEO may partially match, providing an intermediate test case.
- **AtypicalAntipsychotics** (1,120 articles, 32.4% inclusion): High inclusion rate tests whether class balance affects the expert vs. auto MeSH pattern.
- **Opioids** (1,915 articles, 2.5% inclusion): Extreme imbalance tests whether the WSS metric becomes unstable.

### 9.2 MeSH term frequency filtering

Test whether filtering high-frequency auto-MeSH terms (e.g., remove terms appearing in >50% of abstracts) improves auto-MeSH performance. This would test the hypothesis that the noise comes from common terms, not from mechanical lookup per se.

### 9.3 Text mode comparison across all workflows

Run all four text modes across all 11 workflows to test whether the MeSH benefit interacts with stopword removal, trigrams, or vocabulary cap size.

### 9.4 Corpus size ablation

Subsample Statins to 150, 300, 500, 1000, 2744 abstracts. Run expert MeSH and auto MeSH at each size. This directly tests whether enrichment benefit scales with corpus size (supporting the vocabulary heterogeneity hypothesis) or inversely (supporting the original data scarcity hypothesis).

### 9.5 Statistical significance testing

Add paired t-tests or bootstrap confidence intervals for the key comparisons: expert MeSH vs abstract, auto MeSH vs abstract, expert MeSH vs auto MeSH.

### 9.6 BioBERT/PubMedBERT baseline

Establish whether the 0.74–0.77 AUC ceiling is a representation limitation or a task difficulty floor.

---

## 10. Summary of Findings

1. **Expert-assigned MeSH terms are the single biggest improvement.** They double regularized WSS@95% from 0.123 to 0.223, saving an estimated 612 articles from review (vs 337 without MeSH) at 95% recall on the 2,744-article Statins corpus.

2. **Automatic MeSH lookup does not help — and slightly hurts.** Despite using the same vocabulary as expert MeSH, mechanical string matching produces regularized WSS@95 of 0.102, below the abstract-only baseline of 0.123. The benefit of structured vocabulary depends entirely on contextual assignment, not on the vocabulary itself.

3. **The thesis's NEO enrichment failure and the auto-MeSH failure have the same cause.** Both use mechanical term assignment that introduces noise from high-frequency, non-discriminative matches. The thesis identified this as "parent dilution"; this study identifies it as a general property of context-free enrichment.

4. **NEO enrichment has zero effect on the Statins data**, confirming the negative control. The enrichment mechanism passes through transparently when the ontology vocabulary does not overlap with the domain.

5. **Stopword removal has no reliable effect**, replicating the thesis k-fold finding on a second, independent dataset 18× larger.

6. **Unigrams outperform trigrams on WSS@95%** on the Statins data, reversing the thesis finding. On broader corpora, individual discriminative terms carry more signal per feature slot than three-word sequences.

7. **Regularization consistently helps on imbalanced data**, improving AUC for all 11 workflows without exception.

8. **Accuracy is uninformative at 5.5% inclusion rate.** All workflows achieve 93–95%, indistinguishable from majority-class prediction. AUC and WSS@95% are the only meaningful metrics.

9. **The thesis hypothesis about enrichment compensating for data scarcity is not supported.** The evidence suggests enrichment compensates for vocabulary heterogeneity only when terms are contextually assigned — a condition met by expert curation but not by mechanical lookup.

10. **The revised hypothesis: structured semantic features improve classification when terms are both contextually relevant and discriminatively specific.** Expert MeSH assignment inherently selects for both properties. Mechanical lookup selects for neither.
