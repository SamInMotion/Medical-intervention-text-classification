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
