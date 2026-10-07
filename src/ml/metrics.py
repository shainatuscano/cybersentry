"""Metric helpers shared by training and test evaluation."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_recall_fscore_support)


def summarize(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], benign_idx: int = 0):
    """Return (summary dict, per-class DataFrame).

    macro_f1 is averaged over the classes that actually occur in y_true, so a class that is
    missing from a split does not drag the score down.
    """
    k = len(classes)
    present = np.unique(y_true)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(k)), zero_division=0)
    per = pd.DataFrame({"class": classes, "precision": p, "recall": r, "f1": f, "support": s})
    attack_true, attack_pred = y_true != benign_idx, y_pred != benign_idx
    n_attack, n_benign = attack_true.sum(), (~attack_true).sum()
    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=present, average="weighted", zero_division=0),
        "attack_detection_rate": (attack_true & attack_pred).sum() / max(n_attack, 1),
        "false_positive_rate": ((~attack_true) & attack_pred).sum() / max(n_benign, 1),
    }
    return summary, per


def plot_confusion(y_true, y_pred, classes, path, title):
    k = len(classes)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(k)))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(k), classes, rotation=45, ha="right")
    ax.set_yticks(range(k), classes)
    for i in range(k):
        for j in range(k):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Share of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
