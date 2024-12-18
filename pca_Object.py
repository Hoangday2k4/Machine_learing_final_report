import numpy as np
import pandas as pd

class PCA_Object:
    def __init__(self, n_components):
        """
        Khởi tạo class PCA.

        Parameters:
        - n_components (int): Số lượng thành phần chính cần giữ lại.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X):
        """
        Huấn luyện PCA trên dữ liệu đầu vào.

        Parameters:
        - X (pd.DataFrame or np.ndarray): Dữ liệu đầu vào, với các cột là các đặc trưng.
        """
        # Chuyển đổi X thành numpy array nếu là Pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Tính trung bình của từng đặc trưng
        self.mean = np.mean(X, axis=0)

        # Chuẩn hóa dữ liệu bằng cách trừ đi trung bình
        X_centered = X - self.mean

        # Tính ma trận hiệp phương sai
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # Tính toán các giá trị riêng và vector riêng
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sắp xếp giá trị riêng và vector riêng theo thứ tự giảm dần
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Giữ lại các thành phần chính theo n_components
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]

    def transform(self, X):
        """
        Biến đổi dữ liệu đầu vào sang không gian thành phần chính.

        Parameters:
        - X (pd.DataFrame or np.ndarray): Dữ liệu đầu vào.

        Returns:
        - np.ndarray: Dữ liệu đã được giảm chiều.
        """
        # Chuyển đổi X thành numpy array nếu là Pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Chuẩn hóa dữ liệu
        X_centered = X - self.mean

        # Chiếu dữ liệu lên không gian thành phần chính
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """
        Kết hợp huấn luyện và biến đổi dữ liệu.

        Parameters:
        - X (pd.DataFrame or np.ndarray): Dữ liệu đầu vào.

        Returns:
        - np.ndarray: Dữ liệu đã được giảm chiều.
        """
        self.fit(X)
        return self.transform(X)

    def explained_variance_ratio(self):
        """
        Tính tỷ lệ phương sai giải thích bởi các thành phần chính.

        Returns:
        - np.ndarray: Tỷ lệ phương sai giải thích.
        """
        total_variance = np.sum(self.explained_variance)
        return self.explained_variance / total_variance