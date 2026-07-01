# BiomedBERT token-length distribution per Cohen topic

Tokenizer: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`. Truncation cap: 512 subword tokens. Counts include `[CLS]` and `[SEP]`.

| Topic | Mode | n | Median | p95 | Max | Truncated@512 |
|---|---|---|---|---|---|---|
| Statins | abstract | 2744 | 312 | 531 | 1929 | 167 (6.1%) |
| Statins | title_abstract | 2744 | 331 | 555 | 1947 | 218 (7.9%) |
| Statins | title_abstract_mesh | 2744 | 381 | 611 | 1982 | 415 (15.1%) |
| Opiods | abstract | 1772 | 291 | 506 | 1075 | 83 (4.7%) |
| Opiods | title_abstract | 1772 | 308 | 527 | 1095 | 103 (5.8%) |
| Opiods | title_abstract_mesh | 1772 | 357 | 578 | 1163 | 184 (10.4%) |
| ADHD | abstract | 803 | 294 | 497 | 986 | 37 (4.6%) |
| ADHD | title_abstract | 803 | 313 | 524 | 1002 | 44 (5.5%) |
| ADHD | title_abstract_mesh | 803 | 363 | 582 | 1046 | 95 (11.8%) |

## Reading

The truncated@512 column is the direct answer to the part of Christer's Q1 that could read as 'is BERT seeing less of each abstract than BoW?' BoW operates on the full token sequence; BERT loses everything beyond 512 subword tokens. If the truncation rate is under ~10% in expert-MeSH mode (the mode where the gap lives in the BoW pipeline), this is a small effect and worth a one-sentence disclosure in §3.5. If it climbs above 25%, it needs a paragraph in §6 limitations, framing it as a meaningful asymmetry between the two classifiers.
