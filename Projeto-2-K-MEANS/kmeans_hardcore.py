import numpy as np


class KMeansHardcore:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.inertia_ = None

    def fit(self, X):
        np.random.seed(42)
        random_idx = np.random.permutation(X.shape[0])[: self.n_clusters]
        self.centroids = X[random_idx].astype(float)

        for _ in range(self.max_iter):
            distances = self._euclidean_distance(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)

            new_centroids = np.array(
                [
                    X[self.labels == j].mean(axis=0)
                    if np.any(self.labels == j)
                    else self.centroids[j]
                    for j in range(self.n_clusters)
                ]
            )

            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break

            self.centroids = new_centroids

        self.inertia_ = self._compute_inertia(X)
        return self

    def predict(self, X):
        distances = self._euclidean_distance(X, self.centroids)
        return np.argmin(distances, axis=1)

    def _euclidean_distance(self, X, centroids):
        return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    def _compute_inertia(self, X):
        # WCSS: soma das distancias quadradas de cada ponto ao seu centroide
        inertia = 0.0
        for j in range(self.n_clusters):
            points = X[self.labels == j]
            if len(points) == 0:
                continue
            inertia += np.sum((points - self.centroids[j]) ** 2)
        return float(inertia)
