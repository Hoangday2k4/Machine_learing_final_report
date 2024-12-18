import numpy as np
import pandas as pd

class LDA_Object:
    def __init__(self, n_components):
        """
        Khởi tạo class LDA.

        Parameters:
        - n_components (int): Số lượng thành phần cần giữ lại.
        """
        self.n_components = n_components
        self.means = None
        self.scalings = None

    def fit(self, X, y):
        """
        Huấn luyện LDA trên dữ liệu đầu vào.

        Parameters:
        - X (pd.DataFrame or np.ndarray): Dữ liệu đầu vào, với các cột là các đặc trưng.
        - y (pd.Series or np.ndarray): Nhãn tương ứng với các hàng của X.
        """
        # Chuyển đổi X và y thành numpy array nếu là Pandas
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Tính toán các trung bình theo từng lớp
        n_features = X.shape[1]
        classes = np.unique(y)
        overall_mean = np.mean(X, axis=0)

        # Tính scatter trong lớp và scatter giữa các lớp
        S_W = np.zeros((n_features, n_features))  # Within-class scatter matrix
        S_B = np.zeros((n_features, n_features))  # Between-class scatter matrix

        for c in classes:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            S_W += np.dot((X_c - mean_c).T, (X_c - mean_c))
            n_c = X_c.shape[0]
            mean_diff = (mean_c - overall_mean).reshape(-1, 1)
            S_B += n_c * np.dot(mean_diff, mean_diff.T)

        # Tính các vector riêng và giá trị riêng từ ma trận S_W^-1 * S_B
        eigvals, eigvecs = np.linalg.eigh(np.linalg.pinv(S_W).dot(S_B))

        # Sắp xếp giá trị riêng và vector riêng theo thứ tự giảm dần
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]

        # Lưu các vector riêng tương ứng với các thành phần giữ lại
        self.scalings = eigvecs[:, :self.n_components]

    def transform(self, X):
        """
        Biến đổi dữ liệu đầu vào sang không gian LDA.

        Parameters:
        - X (pd.DataFrame or np.ndarray): Dữ liệu đầu vào.

        Returns:
        - np.ndarray: Dữ liệu đã được giảm chiều.
        """
        # Chuyển đổi X thành numpy array nếu là Pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Chiếu dữ liệu lên không gian LDA
        return np.dot(X, self.scalings)

    def fit_transform(self, X, y):
        """
        Kết hợp huấn luyện và biến đổi dữ liệu.

        Parameters:
        - X (pd.DataFrame or np.ndarray): Dữ liệu đầu vào.
        - y (pd.Series or np.ndarray): Nhãn tương ứng với các hàng của X.

        Returns:
        - np.ndarray: Dữ liệu đã được giảm chiều.
        """
        self.fit(X, y)
        return self.transform(X)