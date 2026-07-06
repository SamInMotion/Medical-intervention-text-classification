"""Run the classification pipeline on Cohen et al. (2006) benchmark data.

Usage:
    python -m src.cohen_pipeline --topic Statins --email you@example.com
    python -m src.cohen_pipeline --topic Statins --email you@example.com --text-mode auto_mesh
    python -m src.cohen_pipeline --topic Statins --email you@example.com --compare-text-modes
    python -m src.cohen_pipeline --topic Statins --email you@example.com --all-workflows
    python -m src.cohen_pipeline --topic Statins --email you@example.com --compare-text-modes --output-file outputs/text_mode_comparison.txt
    python -m src.cohen_pipeline --list-topics

This is a separate entry point from pipeline.py rather than an extension
of its CLI. The thesis pipeline and the benchmark pipeline share the same
preprocessing, vectorization, and model code, but their data loading and
reporting differ enough that cramming both into one argparse interface
would make the code harder to read.
"""

import argparse
import logging
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .config import WORKFLOWS, NUMPY_SEED, TF_SEED, SPLIT_SEED
from .preprocessing import preprocess_corpus
from .features import build_vectorizer, vectorize
from .models import build_logistic_model, build_regularized_model
from .evaluation import (
    get_predictions,
    get_probabilities,
    compute_roc_auc,
    compute_wss_at_recall,
)
from .benchmark_loader import (
    COHEN_TOPICS,
    load_cohen_topic,
    list_topics_summary,
)
from .auto_mesh import build_mesh_vocabulary, prepare_auto_mesh_texts

logger = logging.getLogger(__name__)


def _prepare_texts(df, text_mode):
    """Build the text column based on the chosen mode.

    Modes:
        abstract: abstract text only (default, matches thesis pipeline)
        title_abstract: title + abstract concatenated
        title_abstract_mesh: title + abstract + MeSH terms as space-separated tokens

    Returns a list of strings ready for preprocess_corpus.
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


def set_seeds():
    np.random.seed(NUMPY_SEED)
    try:
        import tensorflow as tf
        tf.random.set_seed(TF_SEED)
    except ImportError:
        pass


def _train_and_evaluate(x_train, y_train, x_test, y_test, epochs, batch_size):
    """Train both models on one fold, return metrics and probabilities."""
    model = build_logistic_model(x_train.shape[1])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    score = model.evaluate(x_test, y_test, verbose=0)
    y_pred = get_predictions(model, x_test)
    y_proba = get_probabilities(model, x_test)

    model_reg = build_regularized_model(x_train.shape[1])
    model_reg.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    score_reg = model_reg.evaluate(x_test, y_test, verbose=0)
    y_pred_reg = get_predictions(model_reg, x_test)
    y_proba_reg = get_probabilities(model_reg, x_test)

    return {
        "baseline_acc": score[1],
        "regularized_acc": score_reg[1],
        "y_test": y_test,
        "y_pred_baseline": y_pred,
        "y_pred_regularized": y_pred_reg,
        "y_proba_baseline": y_proba,
        "y_proba_regularized": y_proba_reg,
    }


def run_cohen_kfold(
    topic,
    tsv_path,
    cache_dir,
    email,
    api_key=None,
    text_mode="abstract",
    workflow_id=10,
    n_folds=5,
    epochs=42,
    batch_size=5,
):
    """Run k-fold CV on one Cohen topic with one workflow configuration.

    NEO enrichment is skipped for Cohen data (NEO covers neurological
    examination terms, not drug efficacy). The enrichment workflows
    still run — they just won't find any NEO matches, which is the
    expected negative control documented in cohen_benchmark_design.md.
    """
    set_seeds()
    config = WORKFLOWS[workflow_id]

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

    if text_mode == "auto_mesh":
        mesh_vocab = build_mesh_vocabulary(cache_dir)
        texts = prepare_auto_mesh_texts(df, mesh_vocab)
    else:
        texts = _prepare_texts(df, text_mode)
    labels = np.array(df["labels"])

    # NEO enrichment won't match drug class vocabulary,
    # but we pass None to skip the ontology load entirely
    # when the config doesn't request enrichment
    neo_dict = None
    stopwords = None
    if config.dropstop:
        import nltk
        nltk.download("stopwords", quiet=True)
        stopwords = set(nltk.corpus.stopwords.words("english"))

    processed_texts = preprocess_corpus(texts, config, neo_dict, stopwords)

    print(f"\nCohen topic: {topic} ({text_mode} mode)")
    print(f"Workflow {workflow_id}: {config}")
    print(f"{n_folds}-fold CV on {len(labels)} abstracts ({labels.sum()} included, {(1-labels).sum()} excluded)")
    print("-" * 60)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SPLIT_SEED)

    baseline_accs, reg_accs = [], []
    baseline_aucs, reg_aucs = [], []
    baseline_wss, reg_wss = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(processed_texts, labels)):
        train_texts = [processed_texts[i] for i in train_idx]
        test_texts = [processed_texts[i] for i in test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        tokenizer = build_vectorizer(train_texts, config.nrfeats)
        x_train = vectorize(train_texts, tokenizer)
        x_test = vectorize(test_texts, tokenizer)

        results = _train_and_evaluate(x_train, y_train, x_test, y_test, epochs, batch_size)

        baseline_accs.append(results["baseline_acc"])
        reg_accs.append(results["regularized_acc"])

        b_auc = compute_roc_auc(y_test, results["y_proba_baseline"])
        r_auc = compute_roc_auc(y_test, results["y_proba_regularized"])
        baseline_aucs.append(b_auc)
        reg_aucs.append(r_auc)

        b_wss = compute_wss_at_recall(y_test, results["y_proba_baseline"])
        r_wss = compute_wss_at_recall(y_test, results["y_proba_regularized"])
        baseline_wss.append(b_wss["wss"])
        reg_wss.append(r_wss["wss"])

        print(
            f"  Fold {fold_idx + 1}: "
            f"BL acc={results['baseline_acc']:.3f} AUC={b_auc:.3f} WSS={b_wss['wss']:.3f}  "
            f"Reg acc={results['regularized_acc']:.3f} AUC={r_auc:.3f} WSS={r_wss['wss']:.3f}"
        )

    def _safe_mean_std(vals):
        valid = [v for v in vals if not np.isnan(v)]
        if not valid:
            return float("nan"), float("nan")
        return np.mean(valid), np.std(valid)

    bl_acc_m, bl_acc_s = _safe_mean_std(baseline_accs)
    rg_acc_m, rg_acc_s = _safe_mean_std(reg_accs)
    bl_auc_m, bl_auc_s = _safe_mean_std(baseline_aucs)
    rg_auc_m, rg_auc_s = _safe_mean_std(reg_aucs)
    bl_wss_m, bl_wss_s = _safe_mean_std(baseline_wss)
    rg_wss_m, rg_wss_s = _safe_mean_std(reg_wss)

    print(f"\nBaseline:     acc={bl_acc_m:.3f}+/-{bl_acc_s:.3f}  AUC={bl_auc_m:.3f}+/-{bl_auc_s:.3f}  WSS@95={bl_wss_m:.3f}+/-{bl_wss_s:.3f}")
    print(f"Regularized:  acc={rg_acc_m:.3f}+/-{rg_acc_s:.3f}  AUC={rg_auc_m:.3f}+/-{rg_auc_s:.3f}  WSS@95={rg_wss_m:.3f}+/-{rg_wss_s:.3f}")

    return {
        "topic": topic,
        "text_mode": text_mode,
        "workflow": workflow_id,
        "n_samples": len(labels),
        "n_included": int(labels.sum()),
        "n_folds": n_folds,
        "baseline_acc_mean": bl_acc_m,
        "baseline_acc_std": bl_acc_s,
        "regularized_acc_mean": rg_acc_m,
        "regularized_acc_std": rg_acc_s,
        "baseline_auc_mean": bl_auc_m,
        "baseline_auc_std": bl_auc_s,
        "regularized_auc_mean": rg_auc_m,
        "regularized_auc_std": rg_auc_s,
        "baseline_wss_mean": bl_wss_m,
        "baseline_wss_std": bl_wss_s,
        "regularized_wss_mean": rg_wss_m,
        "regularized_wss_std": rg_wss_s,
    }


def run_cohen_all_workflows(
    topic,
    tsv_path,
    cache_dir,
    email,
    api_key=None,
    text_mode="abstract",
    n_folds=5,
    epochs=42,
    batch_size=5,
):
    """Run all 11 workflows on one Cohen topic."""
    all_results = []
    for wf_id in range(11):
        result = run_cohen_kfold(
            topic=topic,
            tsv_path=tsv_path,
            cache_dir=cache_dir,
            email=email,
            api_key=api_key,
            text_mode=text_mode,
            workflow_id=wf_id,
            n_folds=n_folds,
            epochs=epochs,
            batch_size=batch_size,
        )
        if result:
            all_results.append(result)

    if not all_results:
        print("No results to summarize.")
        return all_results

    print("\n" + "=" * 110)
    print(f"SUMMARY: {topic} ({text_mode}), {n_folds}-fold CV")
    print("=" * 110)
    print(
        f"{'WF':>3}  {'Stop':>4}  {'Syn':>3}  {'Par':>3}  {'Ngr':>3}  {'Feat':>4}  "
        f"{'BL Acc':>12}  {'Reg Acc':>12}  {'BL AUC':>12}  {'Reg AUC':>12}  "
        f"{'BL WSS':>12}  {'Reg WSS':>12}"
    )
    print("-" * 110)

    for r in all_results:
        wf = WORKFLOWS[r["workflow"]]
        print(
            f"{r['workflow']:>3}  "
            f"{'Y' if wf.dropstop else 'N':>4}  "
            f"{'Y' if wf.synonyms else 'N':>3}  "
            f"{'Y' if wf.parents else 'N':>3}  "
            f"{str(wf.ngrams):>3}  "
            f"{wf.nrfeats:>4}  "
            f"{r['baseline_acc_mean']:.3f}+/-{r['baseline_acc_std']:.3f}  "
            f"{r['regularized_acc_mean']:.3f}+/-{r['regularized_acc_std']:.3f}  "
            f"{r['baseline_auc_mean']:.3f}+/-{r['baseline_auc_std']:.3f}  "
            f"{r['regularized_auc_mean']:.3f}+/-{r['regularized_auc_std']:.3f}  "
            f"{r['baseline_wss_mean']:.3f}+/-{r['baseline_wss_std']:.3f}  "
            f"{r['regularized_wss_mean']:.3f}+/-{r['regularized_wss_std']:.3f}"
        )

    return all_results


def run_text_mode_comparison(
    topic,
    tsv_path,
    cache_dir,
    email,
    api_key=None,
    workflow_id=8,
    n_folds=5,
    epochs=42,
    batch_size=5,
):
    """Run one workflow across all four text modes for comparison.

    Default workflow 8 (no stopwords, trigrams, no enrichment) chosen
    because it's the simplest trigram config — isolates the text mode
    effect without enrichment or stopword noise.
    """
    modes = ["abstract", "title_abstract", "title_abstract_mesh", "auto_mesh"]
    results = []

    for mode in modes:
        result = run_cohen_kfold(
            topic=topic,
            tsv_path=tsv_path,
            cache_dir=cache_dir,
            email=email,
            api_key=api_key,
            text_mode=mode,
            workflow_id=workflow_id,
            n_folds=n_folds,
            epochs=epochs,
            batch_size=batch_size,
        )
        if result:
            results.append(result)

    if not results:
        return results

    print("\n" + "=" * 90)
    print(f"TEXT MODE COMPARISON: {topic}, workflow {workflow_id}, {n_folds}-fold CV")
    print("=" * 90)
    print(f"{'Mode':<25}  {'BL Acc':>12}  {'BL AUC':>12}  {'BL WSS@95':>12}  {'Reg Acc':>12}  {'Reg AUC':>12}  {'Reg WSS@95':>12}")
    print("-" * 90)

    for r in results:
        print(
            f"{r['text_mode']:<25}  "
            f"{r['baseline_acc_mean']:.3f}+/-{r['baseline_acc_std']:.3f}  "
            f"{r['baseline_auc_mean']:.3f}+/-{r['baseline_auc_std']:.3f}  "
            f"{r['baseline_wss_mean']:.3f}+/-{r['baseline_wss_std']:.3f}  "
            f"{r['regularized_acc_mean']:.3f}+/-{r['regularized_acc_std']:.3f}  "
            f"{r['regularized_auc_mean']:.3f}+/-{r['regularized_auc_std']:.3f}  "
            f"{r['regularized_wss_mean']:.3f}+/-{r['regularized_wss_std']:.3f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run classification pipeline on Cohen et al. (2006) benchmark data"
    )
    parser.add_argument(
        "--topic", type=str, default="Statins",
        help=f"Drug class topic (default: Statins). Options: {', '.join(COHEN_TOPICS)}",
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
        help="Email for NCBI Entrez API (required by usage policy)",
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument(
        "--text-mode", type=str, default="abstract",
        choices=["abstract", "title_abstract", "title_abstract_mesh", "auto_mesh"],
        help="What text to classify: abstract only, title+abstract, title+abstract+expert MeSH, or abstract+auto-looked-up MeSH",
    )
    parser.add_argument(
        "--workflow", type=int, default=8, choices=range(11),
        help="Workflow config (0-10, default: 8 = trigrams, no enrichment, no stopwords)",
    )
    parser.add_argument("--kfold", type=int, default=5, metavar="K")
    parser.add_argument("--epochs", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument(
        "--all-workflows", action="store_true",
        help="Run all 11 workflows on the selected topic",
    )
    parser.add_argument(
        "--compare-text-modes", action="store_true",
        help="Run abstract vs title+abstract vs expert MeSH vs auto MeSH comparison",
    )
    parser.add_argument(
        "--list-topics", action="store_true",
        help="Print topic summary and exit",
    )
    parser.add_argument(
        "--output-file", type=str, default=None,
        help="Save output to file (in addition to console)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
    )

    # tee output to file if requested
    original_stdout = sys.stdout
    output_file = None
    if args.output_file:
        output_file = open(args.output_file, "w")
        class Tee:
            def __init__(self, *streams):
                self.streams = streams
            def write(self, data):
                for s in self.streams:
                    s.write(data)
            def flush(self):
                for s in self.streams:
                    s.flush()
        sys.stdout = Tee(original_stdout, output_file)

    if args.list_topics:
        summary = list_topics_summary(args.tsv_path)
        print(summary.to_string(index=False))
        return

    if args.compare_text_modes:
        run_text_mode_comparison(
            topic=args.topic,
            tsv_path=args.tsv_path,
            cache_dir=args.cache_dir,
            email=args.email,
            api_key=args.api_key,
            workflow_id=args.workflow,
            n_folds=args.kfold,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    elif args.all_workflows:
        run_cohen_all_workflows(
            topic=args.topic,
            tsv_path=args.tsv_path,
            cache_dir=args.cache_dir,
            email=args.email,
            api_key=args.api_key,
            text_mode=args.text_mode,
            n_folds=args.kfold,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    else:
        run_cohen_kfold(
            topic=args.topic,
            tsv_path=args.tsv_path,
            cache_dir=args.cache_dir,
            email=args.email,
            api_key=args.api_key,
            text_mode=args.text_mode,
            workflow_id=args.workflow,
            n_folds=args.kfold,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    if output_file:
        sys.stdout = original_stdout
        output_file.close()
        print(f"Output saved to {args.output_file}")


if __name__ == "__main__":
    main()
