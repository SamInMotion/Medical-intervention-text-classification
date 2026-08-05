#!/usr/bin/env python
"""Assignment-level analysis of expert versus mechanical MeSH terms.

For each article in a Cohen topic, compares the MeSH terms an NLM indexer
assigned against the terms substring matching recovers from the abstract.

The three quantities of interest:

  recovered  expert terms whose surface form appears in the abstract
  missed     expert terms whose surface form does not appear anywhere in
             the abstract, so mechanical matching cannot recover them
  spurious   vocabulary terms found in the abstract that the indexer did
             not assign to this article

Results are stratified by inclusion label, because WSS@95% depends only on
where the included articles rank.

Run from the repository root:

    python -m paper_experiments.mesh_assignment_analysis --email you@example.com
"""

import argparse
import inspect
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from src.auto_mesh import build_mesh_vocabulary, lookup_mesh_in_text
from src.benchmark_loader import load_cohen_topic


def find_label_column(df):
    for candidate in ("label", "labels", "included", "include", "y", "target"):
        if candidate in df.columns:
            return candidate
    return None


def summarise(records, top_n):
    """Aggregate per-article records into reportable statistics."""
    if not records:
        return {}

    expert_sizes = [r["n_expert_eligible"] for r in records]
    auto_sizes = [r["n_auto"] for r in records]
    recoveries = [r["recovery_rate"] for r in records if r["n_expert_eligible"] > 0]

    missed_counter = Counter()
    spurious_counter = Counter()
    recovered_counter = Counter()
    for r in records:
        missed_counter.update(r["missed"])
        spurious_counter.update(r["spurious"])
        recovered_counter.update(r["recovered"])

    total_eligible = sum(expert_sizes)
    total_recovered = sum(r["n_recovered"] for r in records)
    total_spurious = sum(r["n_spurious"] for r in records)

    return {
        "n_articles": len(records),
        "mean_expert_terms_eligible": round(mean(expert_sizes), 2),
        "mean_expert_terms_below_min_length": round(
            mean([r["n_expert_short"] for r in records]), 2
        ),
        "mean_auto_terms_matched": round(mean(auto_sizes), 2),
        "mean_recovered_per_article": round(
            mean([r["n_recovered"] for r in records]), 2
        ),
        "mean_missed_per_article": round(mean([r["n_missed"] for r in records]), 2),
        "mean_spurious_per_article": round(mean([r["n_spurious"] for r in records]), 2),
        "micro_recovery_rate": round(total_recovered / total_eligible, 4)
        if total_eligible
        else None,
        "macro_recovery_rate": round(mean(recoveries), 4) if recoveries else None,
        "spurious_share_of_matched": round(
            total_spurious / sum(auto_sizes), 4
        )
        if sum(auto_sizes)
        else None,
        "articles_with_zero_recovery": sum(
            1 for r in records if r["n_expert_eligible"] > 0 and r["n_recovered"] == 0
        ),
        "top_missed_terms": missed_counter.most_common(top_n),
        "top_spurious_terms": spurious_counter.most_common(top_n),
        "top_recovered_terms": recovered_counter.most_common(top_n),
    }


def write_report(path, topic, min_length, vocab_size, groups):
    lines = []
    lines.append(f"MeSH assignment analysis: {topic}")
    lines.append("=" * 70)
    lines.append(f"Vocabulary: {vocab_size} unique terms, minimum length {min_length}")
    lines.append("")

    for name, stats in groups.items():
        if not stats:
            continue
        lines.append(f"--- {name} ({stats['n_articles']} articles)")
        lines.append(
            f"  expert terms per article (eligible): "
            f"{stats['mean_expert_terms_eligible']}"
        )
        lines.append(
            f"  expert terms below min length:       "
            f"{stats['mean_expert_terms_below_min_length']}"
        )
        lines.append(
            f"  terms matched by substring search:   "
            f"{stats['mean_auto_terms_matched']}"
        )
        lines.append(
            f"  of which assigned by the indexer:    "
            f"{stats['mean_recovered_per_article']}"
        )
        lines.append(
            f"  expert terms not in the abstract:    "
            f"{stats['mean_missed_per_article']}"
        )
        lines.append(
            f"  matched but not assigned:            "
            f"{stats['mean_spurious_per_article']}"
        )
        lines.append(f"  recovery rate, micro:  {stats['micro_recovery_rate']}")
        lines.append(f"  recovery rate, macro:  {stats['macro_recovery_rate']}")
        lines.append(
            f"  spurious share of all matched terms: "
            f"{stats['spurious_share_of_matched']}"
        )
        lines.append(
            f"  articles where nothing was recovered: "
            f"{stats['articles_with_zero_recovery']}"
        )
        lines.append("")
        lines.append("  most frequently missed expert terms:")
        for term, count in stats["top_missed_terms"]:
            lines.append(f"    {count:5d}  {term}")
        lines.append("")
        lines.append("  most frequent spurious matches:")
        for term, count in stats["top_spurious_terms"]:
            lines.append(f"    {count:5d}  {term}")
        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topic", default="Statins")
    ap.add_argument("--tsv-path", default="data/cohen/epc-ir.clean.tsv")
    ap.add_argument("--cache-dir", default="data/cohen/cache")
    ap.add_argument(
        "--email",
        required=True,
        help="NCBI Entrez address, required by the usage policy",
    )
    ap.add_argument("--min-length", type=int, default=4)
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--out-prefix", default="outputs/mesh_assignment_analysis")
    args = ap.parse_args()

    try:
        df = load_cohen_topic(
            topic=args.topic,
            tsv_path=args.tsv_path,
            cache_dir=args.cache_dir,
            email=args.email,
        )
    except TypeError as exc:
        print("load_cohen_topic rejected these arguments:", exc)
        print("actual signature:", inspect.signature(load_cohen_topic))
        raise SystemExit(1)

    for column in ("texts", "mesh_terms"):
        if column not in df.columns:
            print(f"expected column {column!r} not found")
            print("available columns:", list(df.columns))
            raise SystemExit(1)

    label_col = find_label_column(df)
    if label_col is None:
        print("no inclusion label column found; available:", list(df.columns))
        raise SystemExit(1)

    vocab = build_mesh_vocabulary(args.cache_dir, min_length=args.min_length)

    included, excluded = [], []
    for _, row in df.iterrows():
        abstract = str(row["texts"])
        raw_expert = row.get("mesh_terms") or []

        expert_all = {str(t).lower() for t in raw_expert}
        expert_eligible = {t for t in expert_all if len(t) >= args.min_length}
        n_short = len(expert_all) - len(expert_eligible)

        auto = set(lookup_mesh_in_text(abstract, vocab))

        recovered = sorted(expert_eligible & auto)
        missed = sorted(expert_eligible - auto)
        spurious = sorted(auto - expert_eligible)

        record = {
            "n_expert_eligible": len(expert_eligible),
            "n_expert_short": n_short,
            "n_auto": len(auto),
            "n_recovered": len(recovered),
            "n_missed": len(missed),
            "n_spurious": len(spurious),
            "recovery_rate": len(recovered) / len(expert_eligible)
            if expert_eligible
            else 0.0,
            "recovered": recovered,
            "missed": missed,
            "spurious": spurious,
        }

        if int(row[label_col]) == 1:
            included.append(record)
        else:
            excluded.append(record)

    groups = {
        "included articles": summarise(included, args.top_n),
        "excluded articles": summarise(excluded, args.top_n),
        "all articles": summarise(included + excluded, args.top_n),
    }

    payload = {
        "topic": args.topic,
        "min_length": args.min_length,
        "vocabulary_size": len(vocab),
        "label_column": label_col,
        "groups": groups,
    }

    json_path = f"{args.out_prefix}_{args.topic.lower()}.json"
    txt_path = f"{args.out_prefix}_{args.topic.lower()}.txt"
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(txt_path, args.topic, args.min_length, len(vocab), groups)

    print(f"wrote {json_path}")
    print(f"wrote {txt_path}")


if __name__ == "__main__":
    main()
