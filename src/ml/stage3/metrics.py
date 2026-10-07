"""Evaluation metrics and plots for Stage 3.

Conventions:
- Macro / weighted averages run over the classes present in y_true, so a class absent from a split
  does not count as a zero.
- Per-class FPR = FP / (FP + TN) and FNR = FN / (FN + TP), one class against the rest.
- "Attack" metrics collapse every non-benign class into one: did the model call an attack an attack?
- ROC-AUC and PR-AUC (average precision) use predicted probabilities, one class against the rest,
  and are only computed for classes with both positives and negatives in y_true.
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,  # noqa: E402
                             precision_recall_fscore_support, roc_auc_score)

# Reference categorical palette, slots 1-3 (validated all-pairs for up to three series).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _nan_to_none(x):
    return None if isinstance(x, float) and math.isnan(x) else x


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None,
             classes: list[str], benign_idx: int = 0) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Return (summary, per-class table, confusion matrix with counts)."""
    k = len(classes)
    labels = list(range(k))
    present = [c for c in labels if (y_true == c).any()]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n = cm.sum()

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = n - tp - fp - fn
    with np.errstate(invalid="ignore", divide="ignore"):
        fpr = np.where(fp + tn > 0, fp / (fp + tn), np.nan)
        fnr = np.where(fn + tp > 0, fn / (fn + tp), np.nan)

    roc, pr = np.full(k, np.nan), np.full(k, np.nan)
    if proba is not None:
        for c in present:
            pos = y_true == c
            if pos.all():
                continue
            roc[c] = roc_auc_score(pos, proba[:, c])
            pr[c] = average_precision_score(pos, proba[:, c])

    per = pd.DataFrame({"class": classes, "support": s, "precision": p, "recall": r, "f1": f,
                        "fpr": fpr, "fnr": fnr, "false_negatives": fn, "false_positives": fp,
                        "roc_auc": roc, "pr_auc": pr})
    sub = per.iloc[present]
    w = sub["support"] / sub["support"].sum()

    attack_true, attack_pred = y_true != benign_idx, y_pred != benign_idx
    n_att, n_ben = int(attack_true.sum()), int((~attack_true).sum())
    summary = {
        "n_rows": int(n),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": sub["precision"].mean(),
        "macro_recall": sub["recall"].mean(),
        "macro_f1": sub["f1"].mean(),
        "weighted_precision": (sub["precision"] * w).sum(),
        "weighted_recall": (sub["recall"] * w).sum(),
        "weighted_f1": (sub["f1"] * w).sum(),
        "attack_detection_rate": (attack_true & attack_pred).sum() / n_att if n_att else float("nan"),
        "attack_fnr": (attack_true & ~attack_pred).sum() / n_att if n_att else float("nan"),
        "benign_fpr": (~attack_true & attack_pred).sum() / n_ben if n_ben else float("nan"),
        "attack_false_negatives": int((attack_true & ~attack_pred).sum()),
        "benign_false_positives": int((~attack_true & attack_pred).sum()),
        "macro_roc_auc_ovr": float(np.nanmean(roc[present])) if proba is not None else float("nan"),
        "macro_pr_auc_ovr": float(np.nanmean(pr[present])) if proba is not None else float("nan"),
    }
    if proba is not None and n_att and n_ben:
        score = 1.0 - proba[:, benign_idx]
        summary["attack_roc_auc"] = roc_auc_score(attack_true, score)
        summary["attack_pr_auc"] = average_precision_score(attack_true, score)
    summary = {k_: _nan_to_none(float(v)) for k_, v in summary.items()}
    return summary, per, cm


# ---------------------------------------------------------------- plots
def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_confusion(cm: np.ndarray, classes: list[str], path, title: str) -> None:
    """Row-normalised (share of the true class) with counts printed in each non-empty cell."""
    k = len(classes)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(k), classes, rotation=40, ha="right")
    ax.set_yticks(range(k), classes)
    for i in range(k):
        for j in range(k):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]:,})", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xlabel("Predicted class", color=INK_2)
    ax.set_ylabel("True class", color=INK_2)
    ax.set_title(title, color=INK, loc="left")
    fig.colorbar(im, ax=ax, label="Share of true class", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grouped_bars(table: pd.DataFrame, category: str, series: str, value: str,
                      path, title: str, ylabel: str) -> None:
    """Grouped bars: one group per `category`, one bar per `series` (at most 3 series)."""
    cats = list(dict.fromkeys(table[category]))
    sers = list(dict.fromkeys(table[series]))
    assert len(sers) <= len(SERIES)
    width = 0.8 / len(sers)
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(cats)), 4.2))
    for i, s in enumerate(sers):
        sub = table[table[series] == s].set_index(category).reindex(cats)
        xs = np.arange(len(cats)) + (i - (len(sers) - 1) / 2) * width
        bars = ax.bar(xs, sub[value].fillna(0), width * 0.92, color=SERIES[i], label=s)
        for b, v in zip(bars, sub[value]):
            if pd.notna(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7, color=INK_2)
    tilt = len(cats) >= 6 or max(len(str(c)) for c in cats) > 12
    ax.set_xticks(range(len(cats)), cats, rotation=25 if tilt else 0, ha="right" if tilt else "center")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.set_title(title, color=INK, loc="left")
    _style(ax)
    long_names = max(len(str(s)) for s in sers) > 20
    ax.legend(frameon=False, ncol=1 if long_names else len(sers), loc="upper left",
              bbox_to_anchor=(1.01, 1) if long_names else (0, -0.25 if tilt else -0.12))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
