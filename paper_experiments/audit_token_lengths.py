"""Compute BiomedBERT token-length distribution for each Cohen topic.

Quantifies the BERT 512-token truncation effect across the three topics
analysed in the paper. Used in audit_bow_bert_data_parity.md to address
the part of Christer Q1 that could read as "is BERT seeing less of each
abstract than BoW?"

Outputs:
    paper_experiments/outputs/audit_token_lengths.json
    paper_experiments/outputs/audit_token_lengths.md

Usage (from repo root):
    python paper_experiments/audit_token_lengths.py --email you@example.com
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

from src.benchmark_loader import load_cohen_topic


OUTPUT_DIR = Path("paper_experiments/outputs")
DEFAULT_TSV = "data/epc-ir.clean.tsv"
DEFAULT_CACHE = "data/pubmed_cache"
TOPICS = ("Statins", "Opiods", "ADHD")
BERT_MAX_LEN = 512
MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"


def tokenize_lengths(texts, tokenizer):
    return [len(tokenizer.encode(t, add_special_tokens=True)) for t in texts]


def percentiles(lengths, pcts=(50, 75, 90, 95, 99, 100)):
    if not lengths:
        return {}
    arr = sorted(lengths)
    out = {}
    n = len(arr)
    for p in pcts:
        if p == 100:
            out[f"p{p}"] = arr[-1]
        else:
            idx = max(0, min(n - 1, int(round((p / 100) * (n - 1)))))
            out[f"p{p}"] = arr[idx]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True, help="NCBI Entrez email")
    parser.add_argument("--tsv-path", default=DEFAULT_TSV)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--api-key", default=os.environ.get("NCBI_API_KEY"))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "transformers not installed in this venv. Install with:\n"
            "  pip install transformers",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    results = {}
    for topic in TOPICS:
        print(f"\nTopic: {topic}")
        df = load_cohen_topic(
            tsv_path=args.tsv_path,
            topic=topic,
            cache_dir=args.cache_dir,
            email=args.email,
            api_key=args.api_key,
        )
        # text modes: abstract only, title+abstract, title+abstract+expert MeSH
        for mode_name in ("abstract", "title_abstract", "title_abstract_mesh"):
            texts = []
            for _, row in df.iterrows():
                parts = []
                if mode_name in ("title_abstract", "title_abstract_mesh"):
                    if row.get("title"):
                        parts.append(str(row["title"]))
                parts.append(str(row["texts"]))
                if mode_name == "title_abstract_mesh":
                    mesh = row.get("mesh_terms", [])
                    if mesh:
                        parts.append(" ".join(mesh))
                texts.append(" ".join(parts))

            lengths = tokenize_lengths(texts, tokenizer)
            n_truncated = sum(1 for l in lengths if l > BERT_MAX_LEN)
            pct_truncated = 100.0 * n_truncated / len(lengths) if lengths else 0.0
            pctiles = percentiles(lengths)
            entry = {
                "n_abstracts": len(lengths),
                "mean_tokens": statistics.mean(lengths) if lengths else 0.0,
                "median_tokens": statistics.median(lengths) if lengths else 0.0,
                "max_tokens": max(lengths) if lengths else 0,
                "percentiles": pctiles,
                "n_truncated_at_512": n_truncated,
                "pct_truncated_at_512": pct_truncated,
            }
            print(
                f"  {mode_name:24s} n={len(lengths):4d}  "
                f"median={entry['median_tokens']:.0f}  max={entry['max_tokens']:4d}  "
                f"truncated@512: {n_truncated}/{len(lengths)} ({pct_truncated:.1f}%)"
            )
            results.setdefault(topic, {})[mode_name] = entry

    json_path = OUTPUT_DIR / "audit_token_lengths.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = ["# BiomedBERT token-length distribution per Cohen topic\n"]
    md.append(
        f"Tokenizer: `{MODEL_ID}`. Truncation cap: {BERT_MAX_LEN} subword tokens. "
        "Counts include `[CLS]` and `[SEP]`.\n"
    )
    md.append("| Topic | Mode | n | Median | p95 | Max | Truncated@512 |")
    md.append("|---|---|---|---|---|---|---|")
    for topic in TOPICS:
        if topic not in results:
            continue
        for mode in ("abstract", "title_abstract", "title_abstract_mesh"):
            e = results[topic].get(mode)
            if not e:
                continue
            md.append(
                f"| {topic} | {mode} | {e['n_abstracts']} | "
                f"{e['median_tokens']:.0f} | {e['percentiles'].get('p95', '?')} | "
                f"{e['max_tokens']} | "
                f"{e['n_truncated_at_512']} ({e['pct_truncated_at_512']:.1f}%) |"
            )
    md.append("")
    md.append("## Reading")
    md.append("")
    md.append(
        "The truncated@512 column is the direct answer to the part of "
        "Christer's Q1 that could read as 'is BERT seeing less of each "
        "abstract than BoW?' BoW operates on the full token sequence; BERT "
        "loses everything beyond 512 subword tokens. If the truncation "
        "rate is under ~10% in expert-MeSH mode (the mode where the gap "
        "lives in the BoW pipeline), this is a small effect and worth a "
        "one-sentence disclosure in §3.5. If it climbs above 25%, it "
        "needs a paragraph in §6 limitations, framing it as a meaningful "
        "asymmetry between the two classifiers."
    )

    md_path = OUTPUT_DIR / "audit_token_lengths.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
