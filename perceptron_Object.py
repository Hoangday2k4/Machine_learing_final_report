import numpy as np

class Perceptron_Object:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Khởi tạo Perceptron.

        Parameters:
        - learning_rate (float): Tốc độ học (alpha).
        - n_iterations (int): Số lần lặp tối đa.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Huấn luyện mô hình trên dữ liệu X và nhãn y.

        Parameters:
        - X (np.ndarray): Dữ liệu đầu vào, dạng (n_samples, n_features).
        - y (np.ndarray): Nhãn đầu ra, dạng (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Đảm bảo nhãn y là {0, 1}
        y = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.sign(linear_output)

                if y[idx] * y_predicted <= 0:  # Sai dự đoán
                    self.weights += self.learning_rate * y[idx] * x_i
                    self.bias += self.learning_rate * y[idx]

    def predict(self, X):
        """
        Dự đoán nhãn cho dữ liệu mới X.

        Parameters:
        - X (np.ndarray): Dữ liệu đầu vào, dạng (n_samples, n_features).

        Returns:
        - np.ndarray: Nhãn dự đoán, dạng (n_samples,).
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output > 0, 1, 0)
    
    def fit_predict(self, X, y):
        self.fit(self, X, y)
        return self.predict(self, X)