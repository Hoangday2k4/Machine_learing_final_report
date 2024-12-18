import numpy as np
import pandas as pd

class KMeans_Object:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4, random_state = 42):
        """
        Khởi tạo class KMeans.

        Parameters:
        - n_clusters (int): Số lượng cụm.
        - max_iter (int): Số lần lặp tối đa.
        - tol (float): Ngưỡng thay đổi để dừng thuật toán (tolerance).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.random_state = random_state

    def fit(self, X):
        """
        Huấn luyện KMeans trên dữ liệu đầu vào.

        Parameters:
        - X (pd.DataFrame or np.ndarray): Dữ liệu đầu vào.
        """
        # Chuyển đổi X thành numpy array nếu là Pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Khởi tạo centroids ngẫu nhiên từ các điểm dữ liệu
        np.random.seed(42)
        random_indices = np.random.permutation(X.shape[0])[:self.n_clusters]
        self.centroids = X[random_indices]

        for i in range(self.max_iter):
            # Gán mỗi điểm dữ liệu vào cụm gần nhất
            distances = self._compute_distances(X, self.centroids)
            labels = np.argmin(distances, axis=1)

            # Tính toán lại centroid của từng cụm
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])

            # Kiểm tra hội tụ (nếu centroids thay đổi nhỏ hơn tol)
            if np.all(np.linalg.norm(self.centroids - new_centroids, axis=1) < self.tol):
                break

            self.centroids = new_centroids

        self.labels_ = labels

    def predict(self, X):
        """
        Dự đoán cụm cho dữ liệu mới.

        Parameters:
        - X (pd.DataFrame or np.ndarray): Dữ liệu đầu vào.

        Returns:
        - np.ndarray: Nhãn của các cụm cho từng điểm dữ liệu.
        """
        # Chuyển đổi X thành numpy array nếu là Pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

    def _compute_distances(self, X, centroids):
        """
        Tính khoảng cách Euclidean từ mỗi điểm đến các centroids.

        Parameters:
        - X (np.ndarray): Dữ liệu đầu vào.
        - centroids (np.ndarray): Tập centroids.

        Returns:
        - np.ndarray: Ma trận khoảng cách.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return distances

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)