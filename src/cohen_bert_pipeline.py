"""Run BiomedBERT classifier on Cohen et al. (2006) benchmark data.

Parallel entry point to src/cohen_pipeline.py with BiomedBERT replacing the
bag-of-words logistic regression. Reuses benchmark_loader, auto_mesh, and
evaluation modules from the existing pipeline — only the classifier changes.

Usage:
    # Quick validation: 200-sample subsample, abstract mode, 2 folds, 1 epoch
    python -m src.cohen_bert_pipeline --topic Statins --email you@example.com \\
        --text-mode abstract --subsample 200 --kfold 2 --epochs 1

    # Single mode, full corpus, 5 folds, 3 epochs (default)
    python -m src.cohen_bert_pipeline --topic Statins --email you@example.com \\
        --text-mode title_abstract_mesh --output-file outputs/bert_expert_mesh.txt

    # All four text modes (H-Pub2 experiment)
    python -m src.cohen_bert_pipeline --topic Statins --email you@example.com \\
        --compare-text-modes --output-file outputs/bert_text_modes_statins.txt

This pipeline tests Addendum 1 hypothesis H-Pub2: does the expert vs auto MeSH
WSS@95 gap (Reg WSS 0.223 vs 0.102 in the BoW pipeline) persist when the
classifier is BiomedBERT?

Hardware notes:
    - CPU: one text mode at full Statins (2,744 abstracts), 5-fold, 3 epochs
      runs in ~5-7 hours.
    - GPU: ~5-10x faster. Set --fp16 if your GPU has Tensor Cores (T4+).
    - The --subsample N flag gives a fast end-to-end validation path.
"""

import argparse
import gc
import logging
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from .config import SPLIT_SEED
from .benchmark_loader import COHEN_TOPICS, load_cohen_topic, list_topics_summary
from .auto_mesh import build_mesh_vocabulary, prepare_auto_mesh_texts
from .evaluation import compute_roc_auc, compute_wss_at_recall
from .bert_models import BertConfig, BiomedBertClassifier


logger = logging.getLogger(__name__)


VALID_TEXT_MODES = ("abstract", "title_abstract", "title_abstract_mesh", "auto_mesh")


def _prepare_texts(df, text_mode):
    """Build raw text per record for the chosen mode.

    Mirrors cohen_pipeline._prepare_texts for the non-auto-mesh modes.
    BERT does its own tokenization so we skip preprocess_corpus entirely —
    no stopword removal, no n-grams, no NEO enrichment.
    """
    texts = []
    for _, row in df.iterrows():
        parts = []
        if text_mode in ("title_abstract", "title_abstract_mesh"):
            if row.get("title"):
                parts.append(str(row["title"]))
        parts.append(str(row["texts"]))
        if text_mode == "title_abstract_mesh":
            mesh = row.get("mesh_terms", [])
            if mesh:
                parts.append(" ".join(mesh))
        texts.append(" ".join(parts))
    return texts


def _build_texts_for_mode(df, text_mode, cache_dir):
    """Dispatch to the right text-construction path for the mode."""
    if text_mode == "auto_mesh":
        mesh_vocab = build_mesh_vocabulary(cache_dir)
        return prepare_auto_mesh_texts(df, mesh_vocab)
    return _prepare_texts(df, text_mode)


def _stratified_subsample(df, n, seed):
    """Stratified subsample preserving inclusion rate.

    Cohen Statins is ~5% positive — a non-stratified subsample of 200 could
    miss positives entirely and break the WSS computation.
    """
    if n >= len(df):
        return df

    labels = df["labels"].values
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    pos_rate = len(pos_idx) / len(df)
    n_pos = max(1, int(round(n * pos_rate)))
    n_neg = n - n_pos

    rng = np.random.default_rng(seed)
    pos_sample = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
    neg_sample = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
    chosen = sorted(np.concatenate([pos_sample, neg_sample]).tolist())
    return df.iloc[chosen].reset_index(drop=True)


def _cleanup_between_folds():
    """Free GPU/CPU memory before next fold's model loads."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_bert_kfold(
    topic,
    tsv_path,
    cache_dir,
    email,
    api_key=None,
    text_mode="abstract",
    n_folds=5,
    epochs=3,
    batch_size=8,
    learning_rate=2e-5,
    max_length=512,
    fp16=False,
    class_weight="balanced",
    subsample=None,
    seed=SPLIT_SEED,
):
    """Run k-fold CV with BiomedBERT on one Cohen topic and one text mode.

    Result dict matches cohen_pipeline.run_cohen_kfold's shape for the
    BERT-relevant fields (no baseline/regularized split — BERT is one model).
    """
    df = load_cohen_topic(
        tsv_path=tsv_path,
        topic=topic,
        cache_dir=cache_dir,
        email=email,
        api_key=api_key,
    )

    if len(df) == 0:
        logger.error("No data for topic '%s'. Aborting.", topic)
        return None

    if subsample:
        df = _stratified_subsample(df, subsample, seed)
        logger.info(
            "Subsampled to %d records (%d included, %d excluded)",
            len(df), int(df["labels"].sum()), int((1 - df["labels"]).sum()),
        )

    texts = _build_texts_for_mode(df, text_mode, cache_dir)
    labels = np.array(df["labels"])

    print(f"\nCohen topic: {topic} ({text_mode} mode)")
    print(f"BiomedBERT, {n_folds}-fold CV on {len(labels)} abstracts "
          f"({labels.sum()} included, {(1 - labels).sum()} excluded)")
    print(f"Epochs: {epochs}, batch_size: {batch_size}, lr: {learning_rate}, "
          f"max_length: {max_length}, fp16: {fp16}, cuda: {torch.cuda.is_available()}")
    print(f"Class weighting: {class_weight}")
    print("-" * 70)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    bert_config_template = BertConfig(
        epochs=epochs,
        train_batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        fp16=fp16,
        seed=seed,
        class_weight=class_weight,
    )

    accs, aucs, wsss, fold_times = [], [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        y_train = labels[train_idx].tolist()
        y_test = labels[test_idx]

        t0 = time.time()
        clf = BiomedBertClassifier(bert_config_template)
        clf.fit(train_texts, y_train)
        probs = clf.predict_proba(test_texts)
        elapsed = time.time() - t0

        y_proba = probs[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        acc = float(np.mean(y_pred == y_test))
        auc = compute_roc_auc(y_test, y_proba)
        wss_result = compute_wss_at_recall(y_test, y_proba, target_recall=0.95)
        wss = wss_result["wss"]

        accs.append(acc)
        aucs.append(auc)
        wsss.append(wss)
        fold_times.append(elapsed)

        print(
            f"  Fold {fold_idx + 1}: "
            f"acc={acc:.3f} AUC={auc:.3f} WSS@95={wss:.3f} "
            f"(train+pred {elapsed:.0f}s)"
        )

        _cleanup_between_folds()

    def _safe_mean_std(vals):
        valid = [v for v in vals if not np.isnan(v)]
        if not valid:
            return float("nan"), float("nan")
        return float(np.mean(valid)), float(np.std(valid))

    acc_m, acc_s = _safe_mean_std(accs)
    auc_m, auc_s = _safe_mean_std(aucs)
    wss_m, wss_s = _safe_mean_std(wsss)

    total_time = sum(fold_times)
    print(
        f"\nBiomedBERT:  acc={acc_m:.3f}+/-{acc_s:.3f}  "
        f"AUC={auc_m:.3f}+/-{auc_s:.3f}  WSS@95={wss_m:.3f}+/-{wss_s:.3f}  "
        f"({total_time:.0f}s total across {n_folds} folds)"
    )

    return {
        "topic": topic,
        "text_mode": text_mode,
        "classifier": "BiomedBERT",
        "n_samples": len(labels),
        "n_included": int(labels.sum()),
        "n_folds": n_folds,
        "epochs": epochs,
        "acc_mean": acc_m,
        "acc_std": acc_s,
        "auc_mean": auc_m,
        "auc_std": auc_s,
        "wss_mean": wss_m,
        "wss_std": wss_s,
        "total_seconds": total_time,
    }


def run_bert_text_mode_comparison(
    topic,
    tsv_path,
    cache_dir,
    email,
    api_key=None,
    n_folds=5,
    epochs=3,
    batch_size=8,
    learning_rate=2e-5,
    max_length=512,
    fp16=False,
    class_weight="balanced",
    subsample=None,
    seed=SPLIT_SEED,
):
    """Run all four text modes for the H-Pub2 mechanism-vs-representation test."""
    results = []
    for mode in VALID_TEXT_MODES:
        result = run_bert_kfold(
            topic=topic,
            tsv_path=tsv_path,
            cache_dir=cache_dir,
            email=email,
            api_key=api_key,
            text_mode=mode,
            n_folds=n_folds,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            fp16=fp16,
            class_weight=class_weight,
            subsample=subsample,
            seed=seed,
        )
        if result:
            results.append(result)

    if not results:
        return results

    print("\n" + "=" * 80)
    print(f"BERT TEXT MODE COMPARISON: {topic}, {n_folds}-fold CV, "
          f"{epochs} epochs")
    print("=" * 80)
    print(f"{'Mode':<25}  {'Accuracy':>14}  {'ROC AUC':>14}  {'WSS@95':>14}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['text_mode']:<25}  "
            f"{r['acc_mean']:.3f}+/-{r['acc_std']:.3f}  "
            f"{r['auc_mean']:.3f}+/-{r['auc_std']:.3f}  "
            f"{r['wss_mean']:.3f}+/-{r['wss_std']:.3f}"
        )
    print("-" * 80)
    print("\nFor comparison, the BoW logistic regression result on Statins was:")
    print("  abstract              acc=.752  AUC=.752  WSS@95=.123 (regularized)")
    print("  title_abstract        acc=.760  AUC=.760  WSS@95=.114 (regularized)")
    print("  title_abstract_mesh   acc=.774  AUC=.774  WSS@95=.223 (regularized)")
    print("  auto_mesh             acc=.749  AUC=.749  WSS@95=.102 (regularized)")
    print("(Expert-vs-auto gap: 0.121 WSS@95. H-Pub2 asks: does this persist with BERT?)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run BiomedBERT classifier on Cohen et al. (2006) benchmark data"
    )
    parser.add_argument(
        "--topic", type=str, default="Statins",
        help=f"Drug class topic. Options: {', '.join(COHEN_TOPICS)}",
    )
    parser.add_argument(
        "--tsv-path", type=str, default="data/cohen/epc-ir.clean.tsv",
        help="Path to Cohen TSV file",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="data/cohen/cache",
        help="Directory for cached PubMed records",
    )
    parser.add_argument(
        "--email", type=str, required=True,
        help="Email for NCBI Entrez API",
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument(
        "--text-mode", type=str, default="abstract",
        choices=list(VALID_TEXT_MODES),
        help="Text construction mode",
    )
    parser.add_argument(
        "--compare-text-modes", action="store_true",
        help="Run all four text modes (H-Pub2 experiment)",
    )
    parser.add_argument(
        "--list-topics", action="store_true",
        help="Print topic summary and exit",
    )
    parser.add_argument("--kfold", type=int, default=5, metavar="K")
    parser.add_argument("--epochs", type=int, default=3,
                        help="BERT fine-tuning epochs (default 3)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size (default 8 for CPU; raise on GPU)")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512,
                        help="Token sequence length (BERT-base max is 512)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 (GPU only)")
    parser.add_argument(
        "--class-weight", type=str, default="balanced",
        choices=["none", "balanced"],
        help="Class weighting for imbalanced data (default: balanced). "
             "Cohen topics range from 0.6%% to 32%% inclusion; balanced is the "
             "empirically justified choice. Use 'none' to compare without weighting.",
    )
    parser.add_argument(
        "--subsample", type=int, default=None,
        help="Use a stratified subsample of N records (quick validation)",
    )
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--output-file", type=str, default=None,
        help="Save output to file (in addition to console)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
    )

    # tee output to file if requested (matches cohen_pipeline.py pattern)
    original_stdout = sys.stdout
    output_file = None
    if args.output_file:
        output_file = open(args.output_file, "w", encoding="utf-8")

        class Tee:
            def __init__(self, *streams):
                self.streams = streams

            def write(self, data):
                for s in self.streams:
                    s.write(data)

            def flush(self):
                for s in self.streams:
                    s.flush()

            def isatty(self):
                # Always return False when teeing to a file.
                # ANSI color codes shouldn't end up in the output file,
                # and libraries (e.g. transformers loading report) check
                # this to decide whether to colorize output.
                return False

            def __getattr__(self, name):
                # Fall through to the original stream for anything else
                # libraries might probe (encoding, buffer, etc).
                return getattr(self.streams[0], name)

        sys.stdout = Tee(original_stdout, output_file)

    # CLI uses "none" string; BertConfig wants Python None for no weighting
    resolved_class_weight = None if args.class_weight == "none" else args.class_weight

    try:
        if args.list_topics:
            summary = list_topics_summary(args.tsv_path)
            print(summary.to_string(index=False))
            return

        if args.compare_text_modes:
            run_bert_text_mode_comparison(
                topic=args.topic,
                tsv_path=args.tsv_path,
                cache_dir=args.cache_dir,
                email=args.email,
                api_key=args.api_key,
                n_folds=args.kfold,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_length=args.max_length,
                fp16=args.fp16,
                class_weight=resolved_class_weight,
                subsample=args.subsample,
                seed=args.seed,
            )
        else:
            run_bert_kfold(
                topic=args.topic,
                tsv_path=args.tsv_path,
                cache_dir=args.cache_dir,
                email=args.email,
                api_key=args.api_key,
                text_mode=args.text_mode,
                n_folds=args.kfold,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_length=args.max_length,
                fp16=args.fp16,
                class_weight=resolved_class_weight,
                subsample=args.subsample,
                seed=args.seed,
            )
    finally:
        if output_file:
            sys.stdout = original_stdout
            output_file.close()
            print(f"Output saved to {args.output_file}")


if __name__ == "__main__":
    main()
