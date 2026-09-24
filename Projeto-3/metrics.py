import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (  # noqa: F401 — confusion_matrix reexportada para main.py
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


def calculate_metrics(y_true, y_pred):
    """Calcula e retorna acuracia, precisao e revocacao (weighted)."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    return accuracy, precision, recall


def plot_confusion_matrix(y_true, y_pred, labels, title="Matriz de Confusao", save_path=None):
    """Plota (e opcionalmente salva) a matriz de confusao."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title)
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close()
    return cm
