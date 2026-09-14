import os
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from kmeans_hardcore import KMeansHardcore
from kmeans_sklearn import run_kmeans_sklearn
from metrics import (
    evaluate_model,
    evaluate_labels,
    plot_confusion,
    plot_elbow,
    plot_silhouette_curve,
)
from pca_plot import plot_pca_clusters

OUTPUT_DIR = "saidas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carregar dados Iris (rotulos usados apenas para comparacao posterior)
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Padronizacao: atributos em escalas diferentes (cm) influenciam a distancia euclidiana
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_raw = X.values

K_RANGE = list(range(2, 9))
K_DETAIL = [2, 3, 4, 5]


def run_elbow_and_silhouette(X_data, tag, use_hardcore=True):
    wcss_sk = []
    sil_sk = []
    wcss_hc = []
    sil_hc = []

    print(f"\n===== Metodo do Cotovelo e Silhouette ({tag}) =====")
    print(f"{'K':>3} | {'WCSS-SK':>12} | {'Sil-SK':>8} | {'WCSS-HC':>12} | {'Sil-HC':>8}")

    for k in K_RANGE:
        labels_sk, _, inertia_sk = run_kmeans_sklearn(X_data, k)
        score_sk = evaluate_labels(X_data, labels_sk)
        wcss_sk.append(inertia_sk)
        sil_sk.append(score_sk)

        if use_hardcore:
            model_hard = KMeansHardcore(n_clusters=k)
            model_hard.fit(X_data)
            score_hc = evaluate_labels(X_data, model_hard.labels)
            wcss_hc.append(model_hard.inertia_)
            sil_hc.append(score_hc)
            print(
                f"{k:>3} | {inertia_sk:12.4f} | {score_sk:8.4f} | "
                f"{model_hard.inertia_:12.4f} | {score_hc:8.4f}"
            )
        else:
            print(f"{k:>3} | {inertia_sk:12.4f} | {score_sk:8.4f} | {'-':>12} | {'-':>8}")

    plot_elbow(
        K_RANGE,
        wcss_sk,
        title=f"Elbow Method - Sklearn ({tag})",
        save_path=os.path.join(OUTPUT_DIR, f"elbow_sklearn_{tag}.png"),
    )
    plot_silhouette_curve(
        K_RANGE,
        sil_sk,
        title=f"Silhouette vs K - Sklearn ({tag})",
        save_path=os.path.join(OUTPUT_DIR, f"silhouette_sklearn_{tag}.png"),
    )

    if use_hardcore:
        plot_elbow(
            K_RANGE,
            wcss_hc,
            title=f"Elbow Method - Hardcore ({tag})",
            save_path=os.path.join(OUTPUT_DIR, f"elbow_hardcore_{tag}.png"),
        )
        plot_silhouette_curve(
            K_RANGE,
            sil_hc,
            title=f"Silhouette vs K - Hardcore ({tag})",
            save_path=os.path.join(OUTPUT_DIR, f"silhouette_hardcore_{tag}.png"),
        )

    best_k_sil = K_RANGE[int(np.nanargmax(sil_sk))]
    print(f"Melhor K pelo Silhouette (Sklearn/{tag}): {best_k_sil} (score={max(sil_sk):.4f})")
    return best_k_sil


print("\n===== Efeito da normalizacao (StandardScaler) =====")
print("Comparando WCSS e Silhouette com e sem padronizacao (Sklearn, K=3):")
for name, data in [("sem_normalizacao", X_raw), ("com_normalizacao", X_scaled)]:
    labels, centers, inertia = run_kmeans_sklearn(data, 3)
    score = evaluate_labels(data, labels)
    print(f"  [{name}] WCSS={inertia:.4f} | Silhouette={score:.4f}")

best_k = run_elbow_and_silhouette(X_scaled, "padronizado", use_hardcore=True)
run_elbow_and_silhouette(X_raw, "sem_padronizacao", use_hardcore=False)

print("\n===== Avaliacao detalhada por K (dados padronizados) =====")
for k in K_DETAIL:
    print(f"\n========== K = {k} ==========")

    print("\n[Hardcore]")
    model_hard = KMeansHardcore(n_clusters=k)
    labels_hard, score_hard, time_hard, inertia_hard = evaluate_model(model_hard, X_scaled)
    print(
        f"Silhouette: {score_hard:.4f} | WCSS: {inertia_hard:.4f} | Tempo: {time_hard:.4f}s"
    )
    plot_confusion(
        y,
        labels_hard,
        f"Hardcore K={k}",
        save_path=os.path.join(OUTPUT_DIR, f"confusion_hardcore_k{k}.png"),
    )

    print("\n[Sklearn]")
    start = time.time()
    labels_sk, centers_sk, inertia_sk = run_kmeans_sklearn(X_scaled, k)
    elapsed_sk = time.time() - start
    score_sk = evaluate_labels(X_scaled, labels_sk)
    print(f"Silhouette: {score_sk:.4f} | WCSS: {inertia_sk:.4f} | Tempo: {elapsed_sk:.4f}s")
    plot_confusion(
        y,
        labels_sk,
        f"Sklearn K={k}",
        save_path=os.path.join(OUTPUT_DIR, f"confusion_sklearn_k{k}.png"),
    )

# Visualizacao PCA: K do Silhouette e K=3 (estrutura natural do Iris)
for k_pca in sorted({best_k, 3}):
    labels_sk, centers_sk, _ = run_kmeans_sklearn(X_scaled, k_pca)
    print(f"\n===== PCA - Sklearn K={k_pca} (padronizado) =====")
    plot_pca_clusters(
        X_scaled,
        labels_sk,
        centers_sk,
        2,
        f"PCA (2D) - Sklearn K={k_pca}",
        save_path=os.path.join(OUTPUT_DIR, f"pca_2d_sklearn_k{k_pca}.png"),
    )
    plot_pca_clusters(
        X_scaled,
        labels_sk,
        centers_sk,
        1,
        f"PCA (1D) - Sklearn K={k_pca}",
        save_path=os.path.join(OUTPUT_DIR, f"pca_1d_sklearn_k{k_pca}.png"),
    )

print("\n===== Analise resumida =====")
print(
    "- Os rotulos reais do Iris nao entram no K-Means; servem so para as matrizes de "
    "confusao (comparacao a posteriori)."
)
print(
    "- A padronizacao evita que atributos com maior variancia dominem a distancia "
    "euclidiana."
)
print(
    "- O Elbow (WCSS) e o Silhouette auxiliam a escolha de K; no Iris, K=3 costuma "
    "alinhar-se as tres especies."
)
print(
    "- Limitacoes do K-Means: assume clusters esfericos/convexos, sensivel a "
    "inicializacao e exige escolher K."
)
print(f"\nFiguras salvas em: {OUTPUT_DIR}/")
