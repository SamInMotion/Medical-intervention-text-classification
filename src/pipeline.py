"""End-to-end classification pipeline.

Usage:
    python -m src.pipeline --workflow 10
    python -m src.pipeline --workflow 10 --kfold 5
    python -m src.pipeline --kfold 10 --all-workflows
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

from .config import WORKFLOWS, ProjectPaths, NUMPY_SEED, TF_SEED, SPLIT_SEED
from .data_loader import load_abstracts, load_neo_ontology, load_stopwords
from .preprocessing import preprocess_corpus
from .features import build_vectorizer, vectorize
from .models import build_logistic_model, build_regularized_model
from .evaluation import (
    get_predictions,
    get_probabilities,
    compute_roc_auc,
    print_classification_report,
    plot_confusion_matrix,
    plot_training_history,
)


def set_seeds():
    np.random.seed(NUMPY_SEED)
    try:
        import tensorflow as tf
        tf.random.set_seed(TF_SEED)
    except ImportError:
        pass


def _load_and_preprocess(workflow_id, data_dir):
    """Load data and preprocess for a given workflow. Shared by both pipelines."""
    config = WORKFLOWS[workflow_id]
    paths = ProjectPaths(data_dir=Path(data_dir))

    df = load_abstracts(paths)
    labels = np.array(df["labels"])
    texts = df["texts"].tolist()

    neo_dict = None
    if config.synonyms or config.parents:
        neo_dict = load_neo_ontology(paths)

    stopwords = None
    if config.dropstop:
        stopwords = load_stopwords(paths)

    processed_texts = preprocess_corpus(texts, config, neo_dict, stopwords)
    return processed_texts, labels, config


def _train_and_evaluate(x_train, y_train, x_test, y_test, epochs, batch_size):
    """Train both models on one fold/split, return accuracies, f1 scores, and probabilities."""
    # baseline
    model = build_logistic_model(x_train.shape[1])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    score = model.evaluate(x_test, y_test, verbose=0)
    y_pred = get_predictions(model, x_test)
    y_proba = get_probabilities(model, x_test)

    # regularized
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


def run_kfold(
    workflow_id=10,
    n_folds=5,
    data_dir="data",
    epochs=42,
    batch_size=5,
):
    """Run k-fold cross-validation for a given workflow.

    Fits the vectorizer on training folds only to avoid data leakage.
    Reports mean and std of accuracy across folds.
    """
    set_seeds()

    processed_texts, labels, config = _load_and_preprocess(workflow_id, data_dir)
    n_samples = len(labels)

    print(f"\nWorkflow {workflow_id}: {config}")
    print(f"{n_folds}-fold CV on {n_samples} abstracts")
    print("-" * 50)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SPLIT_SEED)

    baseline_accs = []
    reg_accs = []
    baseline_aucs = []
    reg_aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(processed_texts, labels)):
        train_texts = [processed_texts[i] for i in train_idx]
        test_texts = [processed_texts[i] for i in test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        # fit vectorizer on training fold only
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

        print(f"  Fold {fold_idx + 1}: baseline={results['baseline_acc']:.3f} (AUC {b_auc:.3f})  regularized={results['regularized_acc']:.3f} (AUC {r_auc:.3f})")

    baseline_mean = np.mean(baseline_accs)
    baseline_std = np.std(baseline_accs)
    reg_mean = np.mean(reg_accs)
    reg_std = np.std(reg_accs)

    # filter NaN for AUC (shouldn't happen with stratified folds but just in case)
    valid_b_aucs = [a for a in baseline_aucs if not np.isnan(a)]
    valid_r_aucs = [a for a in reg_aucs if not np.isnan(a)]
    b_auc_mean = np.mean(valid_b_aucs) if valid_b_aucs else float("nan")
    b_auc_std = np.std(valid_b_aucs) if valid_b_aucs else float("nan")
    r_auc_mean = np.mean(valid_r_aucs) if valid_r_aucs else float("nan")
    r_auc_std = np.std(valid_r_aucs) if valid_r_aucs else float("nan")

    print(f"\nBaseline:     {baseline_mean:.3f} +/- {baseline_std:.3f}  (AUC {b_auc_mean:.3f} +/- {b_auc_std:.3f})")
    print(f"Regularized:  {reg_mean:.3f} +/- {reg_std:.3f}  (AUC {r_auc_mean:.3f} +/- {r_auc_std:.3f})")

    return {
        "workflow": workflow_id,
        "n_folds": n_folds,
        "baseline_accs": baseline_accs,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "regularized_accs": reg_accs,
        "regularized_mean": reg_mean,
        "regularized_std": reg_std,
        "baseline_aucs": baseline_aucs,
        "baseline_auc_mean": b_auc_mean,
        "baseline_auc_std": b_auc_std,
        "regularized_aucs": reg_aucs,
        "regularized_auc_mean": r_auc_mean,
        "regularized_auc_std": r_auc_std,
    }


def run_all_workflows_kfold(n_folds=5, data_dir="data", epochs=42, batch_size=5):
    """Run k-fold CV across all 11 workflows and print comparison table."""
    all_results = []
    for wf_id in range(11):
        result = run_kfold(wf_id, n_folds, data_dir, epochs, batch_size)
        all_results.append(result)

    print("\n" + "=" * 90)
    print(f"SUMMARY: {n_folds}-fold CV across all workflows")
    print("=" * 90)
    print(f"{'WF':>3}  {'Stop':>4}  {'Syn':>3}  {'Par':>3}  {'Ngr':>3}  {'Feat':>4}  {'Baseline':>12}  {'Regularized':>12}  {'BL AUC':>12}  {'Reg AUC':>12}")
    print("-" * 90)

    for r in all_results:
        wf = WORKFLOWS[r["workflow"]]
        print(
            f"{r['workflow']:>3}  "
            f"{'Y' if wf.dropstop else 'N':>4}  "
            f"{'Y' if wf.synonyms else 'N':>3}  "
            f"{'Y' if wf.parents else 'N':>3}  "
            f"{str(wf.ngrams):>3}  "
            f"{wf.nrfeats:>4}  "
            f"{r['baseline_mean']:.3f} +/- {r['baseline_std']:.3f}  "
            f"{r['regularized_mean']:.3f} +/- {r['regularized_std']:.3f}  "
            f"{r['baseline_auc_mean']:.3f} +/- {r['baseline_auc_std']:.3f}  "
            f"{r['regularized_auc_mean']:.3f} +/- {r['regularized_auc_std']:.3f}"
        )

    return all_results


def run_pipeline(
    workflow_id=10,
    data_dir="data",
    output_dir="outputs",
    epochs=42,
    batch_size=5,
    show_plots=True,
):
    """Original single-split pipeline for reproducing thesis results."""
    set_seeds()

    config = WORKFLOWS[workflow_id]
    paths = ProjectPaths(data_dir=Path(data_dir), output_dir=Path(output_dir))
    paths.ensure_dirs()

    print(f"Running workflow {workflow_id}: {config}")
    print("=" * 60)

    df = load_abstracts(paths)
    labels = df["labels"]
    texts = df["texts"].tolist()
    print(f"Loaded {len(df)} abstracts ({labels.sum()} included, {(1 - labels).sum()} excluded)")

    neo_dict = None
    if config.synonyms or config.parents:
        neo_dict = load_neo_ontology(paths)

    stopwords = None
    if config.dropstop:
        stopwords = load_stopwords(paths)

    processed_texts = preprocess_corpus(texts, config, neo_dict, stopwords)

    # 60/20/20 split
    from .features import prepare_features
    train_texts, testdev_texts, train_labels, testdev_labels = train_test_split(
        processed_texts, labels,
        test_size=0.4, random_state=SPLIT_SEED, shuffle=True, stratify=labels,
    )
    dev_texts, test_texts, dev_labels, test_labels = train_test_split(
        testdev_texts, testdev_labels,
        test_size=0.5, random_state=SPLIT_SEED, shuffle=True, stratify=testdev_labels,
    )
    print(f"Train: {len(train_labels)}, Dev: {len(dev_labels)}, Test: {len(test_labels)}")

    x_train, x_dev, x_test, tokenizer = prepare_features(
        train_texts, dev_texts, test_texts, config.nrfeats
    )
    y_train = np.array(train_labels)
    y_dev = np.array(dev_labels)
    y_test = np.array(test_labels)
    print(f"Feature matrix: {x_train.shape}")

    # baseline
    print(f"\nTraining baseline ({epochs} epochs)...")
    model = build_logistic_model(x_train.shape[1])
    history = model.fit(
        x_train, y_train, epochs=epochs, batch_size=batch_size,
        validation_data=(x_dev, y_dev), verbose=0,
    )

    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"Baseline test accuracy: {score[1]:.4f}")
    y_pred = get_predictions(model, x_test)
    report = print_classification_report(y_test, y_pred)

    if show_plots:
        plot_training_history(
            history, save_path=paths.output_dir / f"history_wf{workflow_id}.png"
        )
        plot_confusion_matrix(
            y_test, y_pred, save_path=paths.output_dir / f"cm_wf{workflow_id}.png"
        )

    # regularized
    print(f"\nTraining regularized ({epochs} epochs)...")
    model_reg = build_regularized_model(x_train.shape[1])
    history_reg = model_reg.fit(
        x_train, y_train, epochs=epochs, batch_size=32,
        validation_data=(x_dev, y_dev), verbose=0,
    )

    score_reg = model_reg.evaluate(x_test, y_test, verbose=0)
    print(f"Regularized test accuracy: {score_reg[1]:.4f}")
    y_pred_reg = get_predictions(model_reg, x_test)
    report_reg = print_classification_report(y_test, y_pred_reg)

    if show_plots:
        plot_confusion_matrix(
            y_test, y_pred_reg,
            save_path=paths.output_dir / f"cm_reg_wf{workflow_id}.png",
        )

    return {
        "workflow": workflow_id,
        "baseline_accuracy": score[1],
        "baseline_report": report,
        "regularized_accuracy": score_reg[1],
        "regularized_report": report_reg,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run medical abstract classification pipeline"
    )
    parser.add_argument(
        "--workflow", type=int, default=10, choices=range(11),
        help="Workflow config (0-10, default: 10)",
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--kfold", type=int, default=0, metavar="K",
        help="Run K-fold cross-validation instead of single split",
    )
    parser.add_argument(
        "--all-workflows", action="store_true",
        help="Run all 11 workflows (only with --kfold)",
    )
    args = parser.parse_args()

    if args.kfold > 0:
        if args.all_workflows:
            run_all_workflows_kfold(
                n_folds=args.kfold, data_dir=args.data_dir,
                epochs=args.epochs, batch_size=5,
            )
        else:
            run_kfold(
                workflow_id=args.workflow, n_folds=args.kfold,
                data_dir=args.data_dir, epochs=args.epochs, batch_size=5,
            )
    else:
        run_pipeline(
            workflow_id=args.workflow,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            show_plots=not args.no_plots,
        )


if __name__ == "__main__":
    main()
