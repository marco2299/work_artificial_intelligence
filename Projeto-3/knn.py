from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split


def euclidean_distance(point1, point2):
    """Calcula a distancia Euclidiana entre dois pontos."""
    return np.sqrt(np.sum((point1 - point2) ** 2))


def predict_knn(X_train, y_train, X_test, k):
    """Realiza a predicao usando o algoritmo KNN (implementacao manual)."""
    predictions = []
    for test_point in X_test:
        distances = np.linalg.norm(X_train - test_point, axis=1)
        k_indices = np.argsort(distances)[:k]
        k_labels = y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)[0][0]
        predictions.append(most_common)
    return np.array(predictions)


def run_knn(X, y, k=5, dataset_name="Dataset"):
    """
    Divide os dados, executa o KNN e retorna indices, verdadeiros, preditos
    e nome do dataset. Referencia ao Trabalho #01 (KNN).
    """
    X_values = X.values if hasattr(X, "values") else np.asarray(X)
    y_values = y.values if hasattr(y, "values") else np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_values, y_values, test_size=0.2, random_state=42, stratify=y_values
    )
    y_pred = predict_knn(X_train, y_train, X_test, k)
    ids = np.arange(len(y_test))
    return ids, y_test, y_pred, dataset_name
