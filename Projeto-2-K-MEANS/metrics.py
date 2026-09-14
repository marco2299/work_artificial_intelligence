from sklearn.metrics import silhouette_score, confusion_matrix
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, X):
    start = time.time()
    model.fit(X)
    elapsed = time.time() - start
    labels = model.labels
    score = silhouette_score(X, labels) if len(set(labels)) > 1 else float("nan")
    inertia = getattr(model, "inertia_", None)
    return labels, score, elapsed, inertia


def evaluate_labels(X, labels):
    if len(set(labels)) <= 1:
        return float("nan")
    return silhouette_score(X, labels)


def plot_confusion(y_true, y_pred, title="Matriz de Confusao", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Cluster")
    plt.ylabel("Classe Real")
    _finalize_plot(save_path)


def plot_elbow(k_values, wcss_values, title="Metodo do Cotovelo (Elbow)", save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, wcss_values, marker="o")
    plt.xticks(k_values)
    plt.xlabel("Numero de clusters (K)")
    plt.ylabel("Inercia / WCSS")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    _finalize_plot(save_path)


def plot_silhouette_curve(k_values, silhouette_values, title="Silhouette Score vs K", save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, silhouette_values, marker="o", color="green")
    plt.xticks(k_values)
    plt.xlabel("Numero de clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    _finalize_plot(save_path)


def _finalize_plot(save_path=None):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
    # So chama show em backend interativo (evita warning com Agg).
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close()
