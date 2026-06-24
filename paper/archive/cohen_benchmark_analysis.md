# Cohen Benchmark Analysis: Ontology Enrichment Across Datasets and Text Modes

**Author:** Samuel Okoe-Mensah
**Date:** April 14, 2026
**Status:** First complete benchmark run (Statins topic)
**Governing research question:** Can ontology-based feature enrichment improve text classification for systematic review screening when data is scarce?

---

## 1. Purpose

The thesis (University of Bergen, 2021–2024) evaluated ontology-enriched classification on 150 abstracts from one Cochrane systematic review on dementia interventions. The k-fold re-evaluation (April 12, 2026) revised the headline accuracy from 93% to 80–86% and found that NEO enrichment provided no measurable benefit. The thesis examiners and the k-fold analysis both flagged a core limitation: a single domain, a single ontology, and a single dataset size.

This document reports the first benchmark extension: the same 11 workflow configurations tested on the Statins topic from Cohen et al. (2006), a corpus of 2,744 abstracts with a 5.5% inclusion rate. It also reports the first text mode comparison, testing whether adding title text and MeSH terms to the classifier's input improves performance.

**Reference:** Cohen, A.M., Hersh, W.R., Peterson, K., and Yen, P.Y. (2006). Reducing Workload in Systematic Review Preparation Using Automated Citation Classification. *JAMIA*, 13(2), 206–219.

---

## 2. Experimental Setup

### 2.1 Dataset

The Cohen et al. dataset contains 18,733 triage decisions across 15 drug class topics. The Statins topic was selected for the first run because it is the largest topic (3,465 records) and has a low inclusion rate (5.0%), making it representative of real-world screening conditions where most articles are excluded.

After fetching abstracts from PubMed via Biopython Entrez:
- 2 PMIDs could not be retrieved (likely retracted or merged records)
- 719 PMIDs had no abstract text in PubMed (comments, editorials, historical articles, biographical pieces — publication types that do not carry abstracts)
- **Final corpus: 2,744 abstracts, 152 included (5.5%), 2,592 excluded (94.5%)**

The dropout of 721 records (20.8%) is a known consequence of using abstract text only. Cohen et al. used title + abstract + MeSH as features, so they could classify articles without abstracts. The 21 included articles lost to empty abstracts (12.1% of included) is notable but does not materially change the class balance (5.0% → 5.5%).

### 2.2 Pipeline

The same modular pipeline from the thesis v2.0 refactor was used without modification. Logistic regression (single-layer perceptron) with binary bag-of-words features, one-hot encoding, and Keras Tokenizer for vocabulary management. Two model variants: baseline (no regularization) and L2-regularized (lambda = 2e-6). All configurations used 5-fold stratified cross-validation with the vectorizer fitted on training folds only.

### 2.3 NEO enrichment as negative control

The NEO (Neurologic Examination Ontology) covers 1,611 neurological examination concepts. It was designed for the thesis's dementia intervention corpus where terms like "ataxia," "dysmetria," and "tremor" appear frequently. The Statins corpus discusses cholesterol, cardiovascular outcomes, and drug interactions — a vocabulary where NEO terms are almost entirely absent. Workflows 2, 3, 5, 6, 7, and 10 (which request synonym and/or parent enrichment) will find no NEO matches. This is intentional: these workflows serve as a negative control, confirming that the enrichment mechanism does not introduce noise when the ontology is irrelevant.

### 2.4 Text modes

Three text modes were tested:

| Mode | Input to classifier | Rationale |
|------|-------------------|-----------|
| `abstract` | Abstract text only | Matches the thesis pipeline; isolates abstract content |
| `title_abstract` | Title + abstract concatenated | Tests whether title adds discriminative signal |
| `title_abstract_mesh` | Title + abstract + MeSH terms as text tokens | Closest to Cohen et al.'s feature set; tests structured metadata as bag-of-words features |

MeSH (Medical Subject Headings) terms are assigned to PubMed articles by trained indexers at the National Library of Medicine. They are structured vocabulary — controlled terms drawn from a hierarchy of biomedical concepts. Adding them as text tokens to the bag-of-words is a form of feature enrichment analogous to the thesis's NEO enrichment, but using expert-assigned terms rather than ontology-lookup terms.

### 2.5 Evaluation metrics

| Metric | Purpose | Interpretation for this dataset |
|--------|---------|-------------------------------|
| Accuracy | Continuity with thesis | **Uninformative here.** A classifier predicting "excluded" for every article achieves 94.5% accuracy. Accuracy differences of <1% between workflows are not meaningful at 5.5% inclusion rate. |
| ROC AUC | Threshold-independent ranking quality | How well the classifier ranks included articles above excluded ones. Higher AUC means the probability scores separate the classes better, regardless of where the decision threshold is set. |
| WSS@95% | Standard metric in SR screening | The proportion of articles a reviewer can skip while still catching 95% of the included articles, minus the 5% baseline from random sampling. This is the metric that matters for the practical question: does the classifier reduce screening workload? |

---

## 3. Results

### 3.1 All 11 workflows on Statins (abstract mode, 5-fold CV)

| WF | Stop | Syn | Par | Ngr | Feat | BL Acc | Reg Acc | BL AUC | Reg AUC | BL WSS@95 | Reg WSS@95 |
|----|------|-----|-----|-----|------|--------|---------|--------|---------|-----------|------------|
| 0 | N | N | N | N | 1000 | .937±.003 | .945±.002 | .740±.047 | .749±.044 | .148±.103 | .119±.084 |
| 1 | Y | N | N | N | 1000 | .934±.007 | .944±.005 | .722±.063 | .733±.061 | .161±.062 | .160±.055 |
| 2 | Y | Y | N | N | 1000 | .932±.007 | .943±.003 | .722±.063 | .733±.057 | .158±.067 | .152±.046 |
| 3 | Y | Y | Y | N | 1000 | .933±.006 | .943±.004 | .723±.065 | .729±.060 | .166±.071 | .141±.031 |
| 4 | Y | N | N | 3 | 1000 | .934±.005 | .944±.002 | .716±.053 | .726±.054 | .135±.087 | .117±.082 |
| 5 | Y | Y | N | 3 | 1000 | .934±.006 | .944±.004 | .717±.053 | .724±.057 | .134±.087 | .109±.069 |
| 6 | Y | Y | Y | 3 | 1000 | .933±.006 | .943±.004 | .717±.053 | .728±.054 | .130±.081 | .132±.081 |
| 7 | N | Y | Y | 3 | 1000 | .934±.007 | .944±.004 | .741±.038 | .754±.036 | .099±.074 | .118±.072 |
| 8 | N | N | N | 3 | 1000 | .933±.006 | .942±.001 | .745±.039 | .756±.036 | .106±.076 | .111±.078 |
| 9 | N | N | N | 3 | 2000 | .943±.004 | .945±.002 | .739±.052 | .752±.047 | .086±.029 | .105±.010 |
| 10 | Y | Y | Y | 3 | 2000 | .942±.005 | .945±.003 | .707±.058 | .718±.056 | .079±.033 | .082±.034 |

### 3.2 Text mode comparison (workflow 8, Statins, 5-fold CV)

| Text Mode | BL Acc | BL AUC | BL WSS@95 | Reg Acc | Reg AUC | Reg WSS@95 |
|-----------|--------|--------|-----------|---------|---------|------------|
| abstract | .933±.008 | .743±.038 | .096±.073 | .943±.004 | .751±.034 | .129±.084 |
| title_abstract | .935±.006 | .749±.036 | .111±.076 | .942±.004 | .759±.034 | .121±.068 |
| title_abstract_mesh | .941±.004 | .767±.038 | .211±.070 | .945±.005 | .777±.036 | .232±.065 |

### 3.3 Thesis k-fold results for comparison (150 abstracts, dementia, 59/41 split)

| WF | Config summary | BL Acc | Reg Acc |
|----|---------------|--------|---------|
| 0 | Raw unigrams | .853±.050 | .853±.034 |
| 8 | Trigrams, no enrichment | .860±.057 | .853±.058 |
| 3 | Stopwords + synonyms + parents | .807±.044 | .793±.049 |
| 10 | Full enrichment + 2000 features | .833±.082 | .800±.073 |

(AUC and WSS@95 were not computed on the thesis data k-fold run. ROC AUC was added later for workflow 0 only: baseline 0.932±0.027.)

---

## 4. Analysis

### 4.1 Accuracy is uninformative on imbalanced data

The most important methodological point in these results is that accuracy cannot be used to compare workflows on the Statins data. All 11 workflows cluster between 93.2% and 94.5% — a range of 1.3 percentage points. The majority-class baseline (predicting "excluded" for every article) would achieve 94.5%. Several regularized models match or exceed this baseline, which means they may be learning to predict "excluded" more often rather than learning to identify included articles.

This contrasts sharply with the thesis data, where the 59/41 class split made accuracy a reasonable metric. On the thesis data, accuracy ranged from 80.7% to 86.0% — a 5.3pp spread on a dataset where random guessing yields 59%. On the Statins data, the accuracy spread is narrower than the standard deviations, and the metric itself carries almost no information about the classifier's ability to find included articles.

AUC and WSS@95% are the metrics that matter for the rest of this analysis. Accuracy is reported for completeness only.

### 4.2 NEO enrichment has no effect on Cohen data (expected)

Comparing enriched vs. non-enriched workflows on AUC:

| Comparison | Without NEO | With NEO | Difference |
|-----------|-------------|----------|------------|
| WF1 vs WF2 (synonyms) | .722 | .722 | 0.000 |
| WF2 vs WF3 (parents) | .722 | .723 | +0.001 |
| WF4 vs WF5 (syn + trigrams) | .716 | .717 | +0.001 |
| WF5 vs WF6 (parents + trigrams) | .717 | .717 | 0.000 |
| WF8 vs WF7 (syn + par + trigrams) | .745 | .741 | −0.004 |

The differences are all within 0.004 AUC — effectively zero. This confirms the negative control: NEO enrichment neither helps nor hurts when the ontology's vocabulary does not overlap with the corpus. The enrichment mechanism passes through transparently.

The same pattern holds on WSS@95%: no consistent direction, differences well within standard deviations.

This is the expected result and validates the experimental design. If enrichment had somehow improved results on a domain where NEO has no relevant terms, that would indicate a methodological problem (e.g., ontology terms acting as regularization noise that accidentally helps).

### 4.3 Stopword removal has no consistent effect — confirming the thesis k-fold finding

On the thesis data, k-fold showed that stopword removal was a split artifact (WF0 85.3% vs WF1 84.7%). On the Statins data, the same pattern holds:

| Metric | Keep stopwords (WF0) | Remove stopwords (WF1) | Difference |
|--------|---------------------|----------------------|------------|
| BL AUC | .740 | .722 | −0.018 |
| BL WSS@95 | .148 | .161 | +0.013 |

Stopword removal slightly hurts AUC but slightly helps WSS@95. The effects go in opposite directions and are both within the standard deviations. The finding replicates: on both the thesis data and the Cohen data, stopword removal does not produce a reliable improvement.

### 4.4 Trigrams hurt AUC on imbalanced data — a reversal from the thesis

On the thesis data, trigrams were the best-performing feature type (WF8 at 86.0% vs WF0 at 85.3%). On the Statins data, the ranking reverses for AUC:

| Config | BL AUC | BL WSS@95 |
|--------|--------|-----------|
| WF0 (unigrams, no enrichment) | **.740** | **.148** |
| WF8 (trigrams, no enrichment) | .745 | .106 |
| WF1 (unigrams, stopwords removed) | .722 | **.161** |
| WF4 (trigrams, stopwords removed) | .716 | .135 |

On AUC, unigrams and trigrams are roughly comparable (WF0 .740 vs WF8 .745). But on WSS@95 — the metric that matters for screening — unigram workflows consistently outperform their trigram counterparts. WF0 gets .148 WSS@95 vs WF8's .106; WF1 gets .161 vs WF4's .135.

Why would unigrams outperform trigrams on WSS@95? One hypothesis: with 2,744 abstracts and a 1,000-feature vocabulary cap, unigrams capture the most discriminative individual terms (drug names, outcome terms, study design keywords). Trigrams dilute the vocabulary cap with three-word sequences that are less generalizable across the diverse articles in a drug class review. On the thesis's 150 abstracts from one systematic review, the vocabulary was consistent enough that trigrams captured useful phrases. On a broader corpus, individual terms carry more signal per feature slot.

### 4.5 MeSH terms are the single biggest improvement — and the key finding

The text mode comparison on workflow 8 produces the clearest result in the entire study:

| Text Mode | BL AUC | BL WSS@95 | Reg WSS@95 |
|-----------|--------|-----------|------------|
| Abstract only | .743 | .096 | .129 |
| Title + abstract | .749 | .111 | .121 |
| Title + abstract + MeSH | **.767** | **.211** | **.232** |

Adding MeSH terms doubles WSS@95 compared to abstract-only classification. The regularized model with MeSH achieves WSS@95 of 0.232 — meaning a reviewer can skip 23.2% of articles (above the 5% random baseline) while still catching 95% of the included ones. Without MeSH, the reviewer can only skip 12.9%.

In practical terms: on the 2,744-article Statins corpus, the MeSH-enhanced classifier would save a reviewer approximately 637 articles they don't need to read (at 95% recall), compared to 354 without MeSH.

Adding the title alone provides a small improvement over abstract only (AUC +0.006, WSS +0.015 baseline), but the effect is modest. The MeSH terms are doing the heavy lifting.

### 4.6 Why MeSH helps: expert knowledge as feature enrichment

This finding connects directly to the thesis's research question about ontology enrichment. The thesis asked whether adding structured semantic knowledge (NEO synonyms, parent concepts) to the bag-of-words improves classification. On the thesis data, NEO enrichment did not help. On the Cohen data, MeSH enrichment helps substantially.

The difference is not the method — both approaches add structured vocabulary terms to the feature vector. The difference is in the quality and relevance of the structured knowledge:

**NEO enrichment on thesis data:** The NEO ontology provides synonyms ("ataxia" → "dyssynergia") and parent concepts ("ataxia" → "motor coordination disorder"). But the thesis corpus (150 abstracts from one Cochrane review on dementia interventions) was identified through a single search strategy, so the articles already use consistent terminology. Synonym bridging is unnecessary when there is no vocabulary mismatch to bridge. And parent concepts ("pain," "motor disorder") are too broad, matching articles that discuss these concepts in unrelated contexts — the "parent dilution" problem.

**MeSH enrichment on Statins data:** MeSH terms are assigned by professional indexers who read each article and select the most specific applicable terms from a controlled vocabulary. A Statin article about liver toxicity gets MeSH terms like "Chemical and Drug Induced Liver Injury" and "Hydroxymethylglutaryl-CoA Reductase Inhibitors" — terms that precisely capture the article's topic at a level of specificity that bag-of-words from the abstract alone cannot guarantee. Unlike NEO's automatic term lookup (which adds terms whether or not they are contextually relevant), MeSH terms are contextually validated by human judgment.

This suggests a refinement of the thesis hypothesis: **structured semantic knowledge improves classification not when it is merely available (i.e., looked up from an ontology), but when it is contextually relevant to each document.** Expert-assigned terms (MeSH) meet this criterion. Automatic ontology lookup (NEO) does not, because the lookup has no way to assess whether a matched term is relevant in context.

### 4.7 Comparison to Cohen et al. (2006) baselines

Cohen et al. reported WSS@95% per topic using a voting perceptron with binary bag-of-words features (title + abstract + MeSH + publication type) and chi-squared feature selection with 5×2 cross-validation. Their average WSS@95% across all 15 topics was approximately 18.5%.

For the Statins topic specifically, the published WSS@95% values ranged from approximately 0% to 20% depending on the feature selection method. Direct numerical comparison is complicated by differences in classifier (voting perceptron vs. logistic regression), cross-validation scheme (5×2 vs. 5-fold stratified), feature representation (binary + chi-squared vs. binary + vocabulary cap), and the fact that our corpus is 21% smaller due to the empty-abstract dropout.

However, the order of magnitude is comparable. Our best result with title + abstract + MeSH (regularized WSS@95 of 0.232) is within the range of Cohen et al.'s results across topics. The fact that a simple logistic regression with binary features achieves comparable workload reduction to the published baseline suggests that the task's difficulty on the Statins topic is dominated by the data, not the classifier.

### 4.8 High WSS variance across folds

WSS@95 standard deviations are large relative to the means (0.029–0.103). Fold 5 consistently produces near-zero or negative WSS@95 across all workflows and text modes. This means one fold in five has essentially no work savings — the classifier must review nearly all articles to achieve 95% recall.

This fold-level instability is expected. WSS@95% depends on the threshold at which recall crosses 0.95, and with 30 included articles per fold, the threshold is sensitive to the specific articles in each fold. A fold that happens to include hard-to-classify positive examples (articles with atypical abstracts) will push the threshold lower and reduce WSS.

The practical implication: WSS@95% on any single train/test split is unreliable. The k-fold mean is the right summary statistic, but even the mean should be interpreted as "the classifier provides *some* workload reduction, not a *specific* percentage."

### 4.9 Regularization consistently helps on imbalanced data

Unlike the thesis data (where regularization was inconsistent), on the Statins data regularization improves results for nearly every workflow:

| Metric | Workflows where regularization helps | Hurts | No change |
|--------|-------------------------------------|-------|-----------|
| Accuracy | 10 | 1 | 0 |
| AUC | 11 | 0 | 0 |

Regularization improves AUC for all 11 workflows without exception. This makes sense: with 5.5% inclusion rate, the unregularized model is prone to overfitting to the majority class during training. L2 regularization prevents the weights from growing too large in the direction of "always predict excluded," preserving the model's ability to assign higher probabilities to genuinely included articles.

---

## 5. Cross-Dataset Comparison

### 5.1 Summary table

| Condition | Thesis (dementia, 150 abstracts) | Cohen Statins (drug, 2744 abstracts) |
|-----------|--------------------------------|--------------------------------------|
| Class balance | 59% included / 41% excluded | 5.5% included / 94.5% excluded |
| Best BL accuracy | 86.0% (WF8) | 94.3% (WF9) — but uninformative |
| Best BL AUC | 0.932 (WF0, single run) | 0.745 (WF8) |
| Best BL WSS@95 | Not computed | 0.166 (WF3) |
| NEO enrichment effect | None | None (negative control) |
| Stopword effect | None (split artifact) | None |
| Trigram effect | +0.7pp accuracy | +0.005 AUC, −0.04 WSS |
| Regularization effect | Inconsistent | Consistently positive |
| Dominant variance source | Train/test split composition | Fold composition for WSS |

### 5.2 What changes with dataset size

The thesis data had 150 abstracts. The Statins topic has 2,744. The larger dataset does not produce higher AUC — in fact, AUC is substantially lower on Statins (0.74) than on the thesis data (0.93). This is not a dataset size effect; it reflects the fundamentally harder classification problem. The thesis corpus contains articles from one Cochrane review where included and excluded articles discuss different aspects of dementia interventions. The Statins corpus contains articles from a broad drug class review where excluded articles may discuss statins in contexts that overlap heavily with included articles (e.g., a statin pharmacokinetics paper excluded because the study design does not meet the review's criteria).

What does change with size: variance drops substantially. On the thesis data, accuracy standard deviations ranged from 3.4% to 8.2%. On the Statins data, accuracy standard deviations range from 0.1% to 0.8%. AUC standard deviations on Statins (0.036–0.065) are also lower. Larger datasets produce more stable estimates, even if the absolute performance is lower.

### 5.3 The ontology enrichment finding generalizes — but requires qualification

The thesis k-fold found that NEO enrichment did not help on 150 dementia abstracts. The Cohen benchmark confirms that NEO enrichment does not help on 2,744 statins abstracts. But these are different negative results with different explanations:

- **Thesis:** NEO covers the right domain (neurological examination) but the corpus is too small and too homogeneous for vocabulary mismatch to be a problem. Enrichment adds noise without adding signal.
- **Cohen:** NEO covers the wrong domain entirely. There are no neurological examination terms in a cardiovascular drug review. Enrichment passes through transparently — it neither adds noise nor signal.

The important finding is that **MeSH enrichment does help on Cohen data**, which shows that the enrichment method itself (adding structured vocabulary terms to the bag-of-words) is sound. The thesis's null result was not because the method is wrong, but because the conditions for it to work (vocabulary mismatch between documents + relevant ontology coverage) were not met on that particular corpus.

---

## 6. Implications for the Research Question

The thesis asked: *Can ontology-based feature enrichment improve text classification for systematic review screening when data is scarce?*

The combined evidence from the thesis k-fold and the Cohen benchmark suggests a refined answer:

**Ontology-based feature enrichment can improve classification, but only when two conditions are met simultaneously:**

1. **The enrichment terms must be contextually relevant to each document.** Expert-assigned MeSH terms meet this condition because a human indexer verified their relevance. Automatic ontology lookup (the thesis method) does not, because it matches terms based on string overlap without assessing contextual fit.

2. **The corpus must exhibit vocabulary heterogeneity that the enrichment can bridge.** The thesis's 150 abstracts from one Cochrane review had consistent vocabulary — synonym bridging was unnecessary. Broader corpora like Cohen et al.'s drug class reviews, where articles span diverse sub-topics and writing styles, provide more opportunity for structured vocabulary to add signal.

The thesis hypothesis that ontology enrichment compensates for data scarcity was not supported by the evidence. On 150 abstracts, enrichment did not help because the vocabulary was already consistent. On 2,744 abstracts, MeSH enrichment helped because the vocabulary was heterogeneous — but this is the opposite of the scarcity hypothesis (more data, more benefit from enrichment, not less data, more benefit).

A more accurate formulation: **structured semantic features compensate for vocabulary heterogeneity, not for data scarcity per se.** Small datasets from narrow search strategies may have low heterogeneity, making enrichment unnecessary. Large datasets from broad searches have high heterogeneity, making enrichment valuable — but they also have enough data that the enrichment's contribution is modest relative to what the raw text features already provide.

---

## 7. Limitations

### 7.1 Single topic

These results are from the Statins topic only. Statins has a 5.5% inclusion rate and is the largest topic in the Cohen dataset. Results may differ on topics with higher inclusion rates (e.g., AtypicalAntipsychotics at 32.4%) or smaller sizes (e.g., Antihistamines at 310 articles). Running additional topics is planned.

### 7.2 Abstract dropout

719 of 3,465 Statins PMIDs (20.8%) had no abstract in PubMed. Of these, 21 were included articles. The analysis is conducted on the 2,744 articles with available abstracts. Cohen et al. could classify articles without abstracts using title and MeSH features alone. The results here are not directly comparable to Cohen et al.'s per-topic results because of this dropout.

### 7.3 Different classifier from Cohen et al.

Cohen et al. used a voting perceptron with chi-squared feature selection. This study uses logistic regression with vocabulary-cap feature selection. The classifiers are both linear models but differ in training algorithm and feature selection method. Performance differences may reflect classifier choice as well as feature engineering choices.

### 7.4 NEO enrichment was a negative control, not a fair test

NEO is a neurological examination ontology applied to a cardiovascular drug review. The null enrichment result is expected and does not test whether domain-matched ontology enrichment would help on the Statins data. Testing MeSH enrichment as an ontology-based feature (rather than as bag-of-words token injection) would be a fairer comparison, but would require a different integration method.

### 7.5 No statistical significance testing

The analysis reports means and standard deviations but does not run formal hypothesis tests. The WSS@95 standard deviations are large enough that many of the observed differences (e.g., WF0 at .148 vs WF1 at .161) may not be statistically significant. Paired t-tests or bootstrap confidence intervals should be added before drawing firm conclusions about workflow ordering.

### 7.6 Fixed hyperparameters

Training ran for 42 epochs with batch size 5 (baseline) or 32 (regularized), same as the thesis. These were not tuned for the larger Statins dataset. Optimal epochs and batch sizes may differ at 2,744 abstracts vs. 150.

---

## 8. Data Provenance

| Component | Source | Retrieval date |
|-----------|--------|---------------|
| Cohen TSV | https://dmice.ohsu.edu/cohenaa/epc-ir-data/epc-ir.clean.tsv | April 14, 2026 |
| PubMed abstracts | NCBI PubMed via Biopython Entrez | April 14, 2026 |
| MeSH terms | Extracted from PubMed XML records | April 14, 2026 |
| Thesis k-fold data | kfold_analysis.md (this repository) | April 12, 2026 |
| NEO ontology | data/neo.json (this repository, not used for Cohen runs) | 2024 |
| Pipeline code | src/ package v2.1 (this repository) | April 14, 2026 |

### 8.1 Reproducibility

All PubMed records are cached as individual JSON files in `data/cohen/cache/`. Re-running the pipeline will use cached data without hitting the PubMed API, producing identical results. The Cohen TSV is included in the repository at `data/cohen/epc-ir.clean.tsv`. Random seeds are fixed in `src/config.py` (NUMPY_SEED=41, TF_SEED=42, SPLIT_SEED=42).

To reproduce:
```
python -m src.cohen_pipeline --topic Statins --email <your-email> --all-workflows
python -m src.cohen_pipeline --topic Statins --email <your-email> --compare-text-modes
```

---

## 9. Next Steps

### 9.1 Additional topics

Run 2–3 additional Cohen topics to test whether the patterns observed on Statins generalize:
- **ADHD** (851 articles, 9.9% inclusion): Neurological/cognitive domain — closest to the thesis's dementia domain. NEO might match some terms here, providing a partial positive control.
- **AtypicalAntipsychotics** (1,120 articles, 32.4% inclusion): High inclusion rate — tests whether class balance affects the patterns.
- **Opioids** (1,915 articles, 2.5% inclusion): Extreme imbalance — tests classifier behavior at very low inclusion rates.

### 9.2 MeSH enrichment across all workflows

The text mode comparison used workflow 8 only. Running all 11 workflows in `title_abstract_mesh` mode would show whether the MeSH benefit interacts with stopword removal or trigram features.

### 9.3 Corpus size ablation

Subsample the Statins topic to 150, 300, 500, 1000, and 2744 abstracts. Run the same workflow at each size. This directly tests the thesis hypothesis about enrichment compensating for data scarcity: if MeSH enrichment helps more at smaller sizes, the hypothesis is supported. If it helps more at larger sizes (as the current evidence suggests), the hypothesis needs revision.

### 9.4 Statistical significance testing

Add paired t-tests or Wilcoxon signed-rank tests for key comparisons: enriched vs. non-enriched, stopwords vs. no stopwords, unigrams vs. trigrams, and abstract vs. title+abstract+MeSH.

### 9.5 BioBERT/PubMedBERT baseline

Establish whether the 0.74 AUC ceiling on Statins is a representation limitation or a task difficulty floor. If transformer models also achieve ~0.74 AUC, the bottleneck is the data itself (overlapping vocabulary between included and excluded articles in a broad drug class review). If transformers reach 0.85+, the bottleneck is the bag-of-words representation.

---

## 10. Summary of Findings

1. **Accuracy is uninformative at 5.5% inclusion rate.** All workflows achieve 93–95%, indistinguishable from majority-class prediction. AUC and WSS@95% are the metrics that matter.

2. **NEO enrichment has no effect on the Statins corpus**, confirming the negative control. The enrichment mechanism passes through transparently when the ontology does not cover the domain vocabulary.

3. **MeSH terms are the single biggest improvement.** Adding expert-assigned MeSH terms to the bag-of-words doubles WSS@95% from 0.10 to 0.21 (baseline) and from 0.13 to 0.23 (regularized). This is the strongest effect observed across all conditions in this study.

4. **The MeSH finding reframes the thesis's null result.** The enrichment method (adding structured vocabulary to the feature vector) works — but only when the added terms are contextually relevant to each document. Expert-assigned MeSH terms meet this condition; automatic ontology lookup does not.

5. **Stopword removal has no reliable effect**, confirming the thesis k-fold finding on a second, independent dataset.

6. **Unigram features outperform trigrams on WSS@95%**, reversing the thesis finding. On a larger, more heterogeneous corpus, individual discriminative terms provide more signal per feature slot than three-word sequences.

7. **Regularization consistently helps on imbalanced data**, improving AUC for all 11 workflows. This contrasts with the inconsistent regularization effect on the balanced thesis data.

8. **The thesis hypothesis about enrichment compensating for data scarcity is not supported.** The evidence suggests enrichment compensates for vocabulary heterogeneity, which increases (not decreases) with corpus size and search breadth.
