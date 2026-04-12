"""End-to-end classification pipeline.

Usage:
    python -m src.pipeline --workflow 10
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .config import WORKFLOWS, ProjectPaths, NUMPY_SEED, TF_SEED, SPLIT_SEED
from .data_loader import load_abstracts, load_neo_ontology, load_stopwords
from .preprocessing import preprocess_corpus
from .features import prepare_features
from .models import build_logistic_model, build_regularized_model
from .evaluation import (
    get_predictions,
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


def run_pipeline(
    workflow_id=10,
    data_dir="data",
    output_dir="outputs",
    epochs=42,
    batch_size=5,
    show_plots=True,
):
    """Run classification for a given workflow config. Returns results dict."""
    set_seeds()

    config = WORKFLOWS[workflow_id]
    paths = ProjectPaths(data_dir=Path(data_dir), output_dir=Path(output_dir))
    paths.ensure_dirs()

    print(f"Running workflow {workflow_id}: {config}")
    print("=" * 60)

    # load data
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

    # preprocess
    processed_texts = preprocess_corpus(texts, config, neo_dict, stopwords)

    # 60/20/20 split
    train_texts, testdev_texts, train_labels, testdev_labels = train_test_split(
        processed_texts, labels,
        test_size=0.4, random_state=SPLIT_SEED, shuffle=True, stratify=labels,
    )
    dev_texts, test_texts, dev_labels, test_labels = train_test_split(
        testdev_texts, testdev_labels,
        test_size=0.5, random_state=SPLIT_SEED, shuffle=True, stratify=testdev_labels,
    )
    print(f"Train: {len(train_labels)}, Dev: {len(dev_labels)}, Test: {len(test_labels)}")

    # vectorize
    x_train, x_dev, x_test, tokenizer = prepare_features(
        train_texts, dev_texts, test_texts, config.nrfeats
    )
    y_train = np.array(train_labels)
    y_dev = np.array(dev_labels)
    y_test = np.array(test_labels)
    print(f"Feature matrix: {x_train.shape}")

    # baseline model
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

    # regularized model
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
    args = parser.parse_args()

    run_pipeline(
        workflow_id=args.workflow,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        show_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
