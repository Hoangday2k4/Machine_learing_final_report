import numpy as np
import pandas as pd

class DBSCAN_Object:
    def __init__(self, eps=0.5, min_samples=5):
        """
        Khởi tạo class DBSCAN.

        Parameters:
        - eps (float): Bán kính tìm kiếm lân cận (epsilon).
        - min_samples (int): Số lượng điểm tối thiểu để tạo thành một cụm.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        """
        Huấn luyện DBSCAN trên dữ liệu đầu vào.

        Parameters:
        - X (pd.DataFrame or np.ndarray): Dữ liệu đầu vào.
        """
        # Chuyển đổi X thành numpy array nếu là Pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        n_samples = X.shape[0]
        self.labels_ = -1 * np.ones(n_samples)  # Gán tất cả các điểm là noise (-1) ban đầu
        cluster_id = 0

        for i in range(n_samples):
            if self.labels_[i] != -1:
                continue  # Điểm đã được gán cụm

            # Tìm các điểm lân cận
            neighbors = self._region_query(X, i)

            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1  # Noise
            else:
                # Tạo cụm mới
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1

    def _region_query(self, X, index):
        """
        Tìm các điểm trong bán kính eps xung quanh một điểm cụ thể.

        Parameters:
        - X (np.ndarray): Dữ liệu đầu vào.
        - index (int): Chỉ số của điểm hiện tại.

        Returns:
        - list: Danh sách các chỉ số của các điểm lân cận.
        """
        distances = np.linalg.norm(X - X[index], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, index, neighbors, cluster_id):
        """
        Mở rộng cụm bằng cách thêm các điểm lân cận thỏa mãn điều kiện.

        Parameters:
        - X (np.ndarray): Dữ liệu đầu vào.
        - index (int): Chỉ số của điểm hiện tại.
        - neighbors (list): Danh sách các điểm lân cận.
        - cluster_id (int): ID của cụm hiện tại.
        """
        self.labels_[index] = cluster_id
        i = 0

        while i < len(neighbors):
            neighbor_index = neighbors[i]

            if self.labels_[neighbor_index] == -1:  # Nếu điểm là noise
                self.labels_[neighbor_index] = cluster_id

            elif self.labels_[neighbor_index] == -1:  # Nếu chưa được thăm
                self.labels_[neighbor_index] = cluster_id

                new_neighbors = self._region_query(X, neighbor_index)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.unique(np.concatenate((neighbors, new_neighbors)))

            i += 1

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_