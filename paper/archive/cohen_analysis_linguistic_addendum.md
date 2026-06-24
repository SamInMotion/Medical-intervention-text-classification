# Addendum: Linguistic Implications of the Auto-MeSH Experiment

**For insertion into cohen_benchmark_analysis.md v2.0 as Section 4.8 (renumbering subsequent sections)**
**Date:** April 14, 2026

---

## Three Levels of Semantic Representation, Not Two

The thesis framed the research question through the distributional hypothesis (Harris, 1954; Sahlgren, 2008): words that occur in similar contexts have similar meanings, and by extension, documents containing similar words belong to similar categories. Bag-of-words classification operationalises this directly — the classifier learns which words co-occur with inclusion vs. exclusion decisions.

Ontology enrichment, as implemented in the thesis, was framed as introducing **paradigmatic relations** into the feature space. Where distributional features capture syntagmatic patterns (which words appear together), ontology enrichment adds paradigmatic knowledge: "ataxia" and "dyssynergia" are synonyms; "ataxia" is a subtype of "motor coordination disorder." The thesis hypothesis was that this paradigmatic augmentation compensates for data scarcity by bridging vocabulary gaps the classifier cannot learn from limited examples.

The four-way comparison in this study reveals that this two-level framing (distributional vs. distributional + paradigmatic) is incomplete. The results identify three distinct levels of semantic representation, each empirically separable:

### Level 1: Distributional (abstract-only baseline)

The classifier learns from word co-occurrence patterns in abstract text. AUC: 0.752 (regularized). WSS@95: 0.123.

This is the Harris (1954) distributional hypothesis in direct operation. The classifier has no access to meaning — it learns statistical associations between word presence and class labels.

### Level 2: Distributional + paradigmatic (auto-MeSH / NEO enrichment)

The classifier receives abstract text augmented with mechanically-matched structured vocabulary terms. Auto-MeSH appends MeSH terms found by string search in the abstract. NEO enrichment appends synonyms and parent concepts looked up from the ontology.

**This level does not improve over Level 1.** Auto-MeSH: AUC 0.749, WSS@95 0.102 (both below Level 1). NEO enrichment on the thesis data: no measurable benefit. NEO on Statins data: zero effect (negative control).

The reason this level fails connects to a property of paradigmatic relations that the thesis did not account for. Paradigmatic substitution (replacing "ataxia" with "dyssynergia") is meaning-preserving in linguistic theory, but in a bag-of-words classifier, it is not feature-preserving. Adding "dyssynergia" to a document that contains "ataxia" does not help the classifier unless "dyssynergia" independently correlates with the inclusion label — which requires it to appear in the training data with informative frequency. On a small, homogeneous corpus, synonyms that don't appear in the original text are unlikely to appear in other documents either. On a larger corpus, high-frequency paradigmatic terms (auto-MeSH matches like "drug therapy," "humans") appear everywhere and carry no discriminative signal.

In both cases — small corpus with NEO, large corpus with auto-MeSH — the paradigmatic enrichment adds terms that are either absent from the training vocabulary (and therefore invisible to the classifier) or present but uninformative (and therefore noise). The enrichment is semantically valid (the paradigmatic relations are real) but statistically useless (the classifier cannot exploit them).

### Level 3: Distributional + conceptual (expert MeSH)

The classifier receives abstract text augmented with human-assigned MeSH terms. These terms are selected by professional indexers who read the article and annotate what it is *about* — its topics, methods, populations, and outcomes — using a controlled vocabulary.

**This level improves substantially over Levels 1 and 2.** AUC: 0.774 (+0.022 over Level 1). WSS@95: 0.223 (+0.100 over Level 1, +0.121 over Level 2).

Expert MeSH terms differ from paradigmatic enrichment in kind, not just in quality. They are not synonyms or hypernyms of words in the text. They are **topic annotations** — onomasiological labels that identify the document's conceptual content. The MeSH term "Hydroxymethylglutaryl-CoA Reductase Inhibitors" assigned to a statin article does not replace or augment any word in the abstract. It adds a new piece of information: an expert has determined that this article belongs to the conceptual category of HMG-CoA reductase inhibitor research.

This distinction maps onto a well-established contrast in lexical semantics:

- **Semasiological** approach: start from the linguistic form (a word), find its meanings (synonyms, hypernyms, definitions). NEO enrichment and auto-MeSH are semasiological — they start from words in the text and look up related forms.
- **Onomasiological** approach: start from the concept (a meaning or topic), find appropriate linguistic expressions. Expert MeSH is onomasiological — the indexer starts from the question "what is this article about?" and assigns the controlled-vocabulary term that best expresses the concept, regardless of what words actually appear in the abstract.

The empirical finding is that onomasiological enrichment (expert MeSH) helps, while semasiological enrichment (auto-MeSH, NEO) does not. The classifier does not need more words that mean the same thing (paradigmatic augmentation). It needs information about what each document means at the conceptual level (topic annotation).

### Implications for the distributional hypothesis debate

The distributional hypothesis holds that meaning is captured by patterns of use. The thesis asked whether distributional features are sufficient for text classification or need supplementation with structured knowledge. The results suggest a more specific answer:

**Distributional features are sufficient to capture word-level meaning** for this task. The classifier does not benefit from knowing that "ataxia" means the same as "dyssynergia" (paradigmatic knowledge) because both words carry the same distributional signal — they appear in the same contexts.

**Distributional features are not sufficient to capture document-level meaning.** The classifier benefits from expert topic annotations because these encode information about the document's content that may not be expressed in the surface vocabulary. An article about statin safety might use words like "adverse events," "hepatotoxicity," and "myalgia" without ever containing the string "Hydroxymethylglutaryl-CoA Reductase Inhibitors." The expert MeSH term adds a conceptual label that links this article to others in the same topic space — a link the distributional features alone cannot make.

This is consistent with Lenci's (2018) argument that distributional semantics captures graded, context-sensitive similarity rather than the crisp categorical relations encoded in ontologies. The ontology's paradigmatic structure (synonymy, hypernymy) does not help because the classifier already captures graded similarity through distributional patterns. What helps is categorical information — this article is about X — which is precisely what expert topic annotation provides and what distributional features and mechanical lookup cannot.

### Implications for ontology-based NLP

The finding does not mean ontologies are useless for text classification. It means the value of an ontology depends on how its terms are mapped to documents:

- **Mechanical lookup (string matching, regex, keyword spotting):** produces semasiological enrichment. Empirically unhelpful in this study. The noise from non-discriminative matches cancels any signal.
- **Expert annotation (manual indexing):** produces onomasiological enrichment. Empirically helpful. But expensive and slow — NLM's MeSH indexing takes weeks after publication.
- **Concept linking with disambiguation (MetaMap, ScispaCy, cTAKES):** potentially intermediate. These systems attempt to map text to ontology concepts with contextual disambiguation, approximating expert judgment programmatically. Not tested in this study, but the results predict they would outperform mechanical lookup to the extent that their disambiguation successfully filters irrelevant matches.

The practical question for future work is whether automated concept linking (an approximation of onomasiological annotation) can close the gap between mechanical lookup and expert assignment. If so, ontology enrichment becomes viable at scale without human indexers.

---

## References for linguistic framing

Harris, Z.S. (1954). Distributional structure. *Word*, 10(2-3), 146–162.

Sahlgren, M. (2008). The distributional hypothesis. *Italian Journal of Linguistics*, 20(1), 33–54.

Lenci, A. (2018). Distributional models of word meaning. *Annual Review of Linguistics*, 4, 151–171.

Saussure, F. de (1916/1959). *Course in General Linguistics*. New York: Philosophical Library. [paradigmatic vs syntagmatic relations]

Baldinger, K. (1980). *Semantic Theory: Towards a Modern Semantics*. Oxford: Blackwell. [semasiological vs onomasiological approaches]

Geeraerts, D. (2010). *Theories of Lexical Semantics*. Oxford: OUP. [overview of both traditions]
