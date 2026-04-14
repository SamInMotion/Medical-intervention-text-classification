"""Load and cache Cohen et al. (2006) benchmark dataset.

Parses the EPC-IR TSV file, fetches abstracts and MeSH terms from
PubMed via Biopython Entrez, and caches results locally so the
API is only hit once per PMID.

Reference:
    Cohen, A.M., Hersh, W.R., Peterson, K., and Yen, P.Y. (2006).
    Reducing Workload in Systematic Review Preparation Using
    Automated Citation Classification. JAMIA 13(2):206-219.
"""

import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# all 15 drug class topics in the dataset
COHEN_TOPICS = [
    "ACEInhibitors",
    "ADHD",
    "Antihistamines",
    "AtypicalAntipsychotics",
    "BetaBlockers",
    "CalciumChannelBlockers",
    "Estrogens",
    "NSAIDS",
    "Opiods",
    "OralHypoglycemics",
    "ProtonPumpInhibitors",
    "SkeletalMuscleRelaxants",
    "Statins",
    "Triptans",
    "UrinaryIncontinence",
]


def parse_cohen_tsv(tsv_path: str) -> pd.DataFrame:
    """Parse the EPC-IR TSV into a DataFrame.

    The file has no header row. Columns:
        topic, endnote_id, pmid, abstract_decision, article_decision

    Labels are mapped to binary: 'I' -> 1, everything else -> 0.
    Windows line endings are stripped.
    """
    rows = []
    with open(tsv_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) != 5:
                logger.warning("Skipping malformed row: %s", row)
                continue
            topic = row[0].strip()
            endnote_id = row[1].strip()
            pmid = row[2].strip()
            abstract_decision = row[3].strip()
            article_decision = row[4].strip()
            rows.append({
                "topic": topic,
                "endnote_id": endnote_id,
                "pmid": pmid,
                "abstract_label": 1 if abstract_decision == "I" else 0,
                "article_label": 1 if article_decision == "I" else 0,
                "abstract_decision_raw": abstract_decision,
                "article_decision_raw": article_decision,
            })

    df = pd.DataFrame(rows)
    logger.info(
        "Parsed %d records across %d topics",
        len(df), df["topic"].nunique(),
    )
    return df


def get_topic_data(
    cohen_df: pd.DataFrame,
    topic: str,
    level: str = "abstract",
) -> Tuple[List[str], List[int]]:
    """Extract PMIDs and binary labels for one topic.

    Args:
        cohen_df: output of parse_cohen_tsv
        topic: one of COHEN_TOPICS
        level: 'abstract' or 'article' triage level

    Returns:
        (list of PMIDs, list of labels)
    """
    if topic not in COHEN_TOPICS:
        raise ValueError(
            f"Unknown topic '{topic}'. "
            f"Valid topics: {COHEN_TOPICS}"
        )
    label_col = f"{level}_label"
    subset = cohen_df[cohen_df["topic"] == topic]
    return subset["pmid"].tolist(), subset[label_col].tolist()


def fetch_pubmed_records(
    pmids: List[str],
    cache_dir: str,
    email: str,
    api_key: Optional[str] = None,
    batch_size: int = 200,
    delay: float = 0.4,
) -> Dict[str, dict]:
    """Fetch abstracts and MeSH terms from PubMed, with local caching.

    Each PMID is cached as a JSON file in cache_dir. Already-cached
    PMIDs are loaded from disk without hitting the API.

    Args:
        pmids: list of PubMed IDs to fetch
        cache_dir: directory for cached JSON files
        email: required by NCBI Entrez (usage policy)
        api_key: optional NCBI API key (raises rate limit to 10/sec)
        batch_size: PMIDs per efetch request (max 200 for XML)
        delay: seconds between API calls (0.34s minimum without key)

    Returns:
        dict mapping PMID -> {title, abstract, mesh_terms}
    """
    from Bio import Entrez

    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    records = {}
    to_fetch = []

    for pmid in pmids:
        cached_file = cache_path / f"{pmid}.json"
        if cached_file.exists():
            with open(cached_file, "r") as f:
                records[pmid] = json.load(f)
        else:
            to_fetch.append(pmid)

    if records:
        logger.info("Loaded %d records from cache", len(records))

    if not to_fetch:
        logger.info("All %d records found in cache", len(pmids))
        return records

    logger.info("Fetching %d records from PubMed...", len(to_fetch))

    for batch_start in range(0, len(to_fetch), batch_size):
        batch = to_fetch[batch_start:batch_start + batch_size]
        batch_end = min(batch_start + batch_size, len(to_fetch))
        logger.info(
            "  Batch %d-%d of %d",
            batch_start + 1, batch_end, len(to_fetch),
        )

        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch),
                rettype="xml",
                retmode="xml",
            )
            import xml.etree.ElementTree as ET

            xml_data = handle.read()
            handle.close()

            root = ET.fromstring(xml_data)
            for article_elem in root.findall(".//PubmedArticle"):
                record = _parse_pubmed_xml_article(article_elem)
                if record and record["pmid"] in batch:
                    records[record["pmid"]] = record
                    cached_file = cache_path / f"{record['pmid']}.json"
                    with open(cached_file, "w") as f:
                        json.dump(record, f)

        except Exception as exc:
            logger.error(
                "Failed to fetch batch starting at %d: %s",
                batch_start, exc,
            )
            # continue with next batch rather than crashing
            # partial results are still cached

        if batch_end < len(to_fetch):
            time.sleep(delay)

    fetched_count = len(records) - (len(pmids) - len(to_fetch))
    missed = [p for p in to_fetch if p not in records]
    if missed:
        logger.warning(
            "Could not fetch %d PMIDs: %s",
            len(missed), missed[:10],
        )
    else:
        logger.info("Fetched all %d new records", fetched_count)

    return records


def _parse_pubmed_xml_article(article_elem) -> Optional[dict]:
    """Extract PMID, title, abstract text, and MeSH terms from one PubmedArticle XML element."""
    pmid_elem = article_elem.find(".//PMID")
    if pmid_elem is None:
        return None
    pmid = pmid_elem.text.strip()

    # title
    title_elem = article_elem.find(".//ArticleTitle")
    title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""

    # abstract — may have multiple AbstractText elements (structured abstract)
    abstract_parts = []
    for abs_elem in article_elem.findall(".//AbstractText"):
        label = abs_elem.get("Label", "")
        text = "".join(abs_elem.itertext()).strip()
        if label and text:
            abstract_parts.append(f"{label}: {text}")
        elif text:
            abstract_parts.append(text)
    abstract = " ".join(abstract_parts)

    # MeSH terms
    mesh_terms = []
    for mesh_heading in article_elem.findall(".//MeshHeading"):
        descriptor = mesh_heading.find("DescriptorName")
        if descriptor is not None and descriptor.text:
            mesh_terms.append(descriptor.text.strip())
        for qualifier in mesh_heading.findall("QualifierName"):
            if qualifier.text:
                mesh_terms.append(qualifier.text.strip())

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "mesh_terms": mesh_terms,
    }


def build_topic_dataframe(
    pmids: List[str],
    labels: List[int],
    records: Dict[str, dict],
) -> pd.DataFrame:
    """Combine labels with fetched PubMed data into a single DataFrame.

    PMIDs without fetched abstracts are dropped with a warning.
    The output has columns matching the thesis data_loader format:
    'labels' and 'texts' (abstract text), plus 'pmid', 'title',
    'mesh_terms' for the benchmark extensions.
    """
    rows = []
    missing = 0
    empty_abstract = 0
    for pmid, label in zip(pmids, labels):
        if pmid not in records:
            missing += 1
            continue
        rec = records[pmid]
        if not rec["abstract"]:
            empty_abstract += 1
            continue
        rows.append({
            "pmid": pmid,
            "labels": label,
            "texts": rec["abstract"],
            "title": rec["title"],
            "mesh_terms": rec["mesh_terms"],
        })

    if missing:
        logger.warning("Dropped %d PMIDs (not fetched from PubMed)", missing)
    if empty_abstract:
        logger.warning("Dropped %d PMIDs (empty abstract)", empty_abstract)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        logger.info(
            "Topic DataFrame: %d abstracts (%d included, %d excluded)",
            len(df), df["labels"].sum(), (1 - df["labels"]).sum(),
        )
    else:
        logger.warning("Topic DataFrame is empty after filtering")
    return df


def load_cohen_topic(
    tsv_path: str,
    topic: str,
    cache_dir: str,
    email: str,
    api_key: Optional[str] = None,
    level: str = "abstract",
) -> pd.DataFrame:
    """One-call function: parse TSV, fetch from PubMed, return ready DataFrame.

    This is the main entry point for pipeline integration.

    Args:
        tsv_path: path to epc-ir.clean.tsv
        topic: one of COHEN_TOPICS
        cache_dir: directory for cached PubMed JSON files
        email: NCBI Entrez email (required by usage policy)
        api_key: optional NCBI API key
        level: 'abstract' or 'article' triage level

    Returns:
        DataFrame with columns: pmid, labels, texts, title, mesh_terms
    """
    cohen_df = parse_cohen_tsv(tsv_path)
    pmids, labels = get_topic_data(cohen_df, topic, level=level)
    logger.info("Topic '%s': %d records at %s level", topic, len(pmids), level)

    records = fetch_pubmed_records(
        pmids, cache_dir, email, api_key=api_key,
    )
    return build_topic_dataframe(pmids, labels, records)


def list_topics_summary(tsv_path: str) -> pd.DataFrame:
    """Return a summary of all topics present in the dataset."""
    cohen_df = parse_cohen_tsv(tsv_path)
    summary_rows = []
    for topic in cohen_df["topic"].unique():
        subset = cohen_df[cohen_df["topic"] == topic]
        total = len(subset)
        incl_abstract = int(subset["abstract_label"].sum())
        incl_article = int(subset["article_label"].sum())
        summary_rows.append({
            "topic": topic,
            "total": total,
            "included_abstract": incl_abstract,
            "included_article": incl_article,
            "pct_included_abstract": round(100 * incl_abstract / total, 1) if total > 0 else 0.0,
            "pct_included_article": round(100 * incl_article / total, 1) if total > 0 else 0.0,
        })
    summary = pd.DataFrame(summary_rows)
    return summary
