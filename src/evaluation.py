"""Evaluation: classification reports, confusion matrices, training plots."""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def get_predictions(model, x_test, threshold=0.5):
    y_proba = model.predict(x_test, verbose=0)
    return (y_proba > threshold).astype("int32").flatten()


def get_probabilities(model, x_test):
    return model.predict(x_test, verbose=0).flatten()


def compute_roc_auc(y_true, y_proba):
    """ROC AUC with fallback for single-class folds."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_proba)


def print_classification_report(y_true, y_pred):
    report_str = classification_report(y_true, y_pred)
    print(report_str)
    return classification_report(y_true, y_pred, output_dict=True)


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    heatmap = sns.heatmap(
        cm, annot=True, annot_kws={"size": 15}, cmap="YlGnBu", vmin=0, fmt="d"
    )
    heatmap.set_xlabel("Predicted", fontsize=14)
    heatmap.set_ylabel("Actual", fontsize=14)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_training_history(history, save_path=None):
    history_df = pd.DataFrame(history.history)
    history_df.plot(xlabel="Epoch")
    plt.title("Training History")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def compute_wss_at_recall(y_true, y_proba, target_recall=0.95):
    """Compute WSS at a given recall threshold.

    Sweeps the classification threshold from low to high, finds the
    point where recall >= target_recall, and computes the work saved.

    WSS@R = (TN + FN) / N - (1 - R)

    Args:
        y_true: binary ground truth labels (1=included, 0=excluded)
        y_proba: predicted probabilities for the positive class
        target_recall: recall threshold (default 0.95)

    Returns:
        dict with keys: wss, threshold, achieved_recall, tn, fn, n
        Returns wss=NaN if recall target cannot be met.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    n_positive = y_true.sum()
    n_total = len(y_true)

    if n_positive == 0:
        return {
            "wss": float("nan"),
            "threshold": float("nan"),
            "achieved_recall": float("nan"),
            "tn": 0, "fn": 0, "n": n_total,
        }

    sorted_indices = np.argsort(-y_proba)
    y_sorted = y_true[sorted_indices]

    cumulative_positives = np.cumsum(y_sorted)
    recall_at_k = cumulative_positives / n_positive

    meeting_target = np.where(recall_at_k >= target_recall)[0]
    if len(meeting_target) == 0:
        return {
            "wss": float("nan"),
            "threshold": float("nan"),
            "achieved_recall": recall_at_k[-1] if len(recall_at_k) > 0 else 0.0,
            "tn": 0, "fn": 0, "n": n_total,
        }

    k = meeting_target[0]
    docs_reviewed = k + 1
    not_reviewed_labels = y_sorted[docs_reviewed:]
    fn = not_reviewed_labels.sum()
    tn = len(not_reviewed_labels) - fn

    achieved_recall = cumulative_positives[k] / n_positive
    threshold = y_proba[sorted_indices[k]]

    wss = (tn + fn) / n_total - (1 - target_recall)

    return {
        "wss": wss,
        "threshold": float(threshold),
        "achieved_recall": float(achieved_recall),
        "tn": int(tn),
        "fn": int(fn),
        "n": n_total,
    }    
