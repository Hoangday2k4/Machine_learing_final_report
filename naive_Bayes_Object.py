import numpy as np
import pandas as pd

class NaiveBayes_Object:
    def __init__(self):
        """
        Class cha Naive Bayes.
        """
        pass

    def fit(self, X, y):
        """
        Huấn luyện mô hình với dữ liệu X và nhãn y.

        Parameters:
        - X (pd.DataFrame or np.ndarray): Dữ liệu đầu vào.
        - y (pd.Series or np.ndarray): Nhãn đầu ra.
        """
        raise NotImplementedError("Phương thức này cần được triển khai bởi lớp con.")

    def predict(self, X):
        """
        Dự đoán nhãn cho dữ liệu mới X.

        Parameters:
        - X (pd.DataFrame or np.ndarray): Dữ liệu đầu vào.

        Returns:
        - np.ndarray: Nhãn dự đoán.
        """
        raise NotImplementedError("Phương thức này cần được triển khai bởi lớp con.")
    def fit_predict(self, X, y):
        """
        Kết hợp huấn luyện và dự đoán nhãn cho dữ liệu.

        """
        raise NotImplementedError("Phương thức này cần được triển khai bởi lớp con.")

class GaussianNB_Object(NaiveBayes_Object):
    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.classes = np.unique(y)
        self.mean = {}
        self.variance = {}
        self.prior = {}

        for cls in self.classes:
            X_cls = X[y == cls]
            self.mean[cls] = X_cls.mean(axis=0)
            self.variance[cls] = X_cls.var(axis=0)
            self.prior[cls] = X_cls.shape[0] / X.shape[0]

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            posteriors = []
            for cls in self.classes:
                prior = np.log(self.prior[cls])
                conditional = -0.5 * np.sum(np.log(2 * np.pi * self.variance[cls])) - 0.5 * np.sum(((x - self.mean[cls]) ** 2) / (self.variance[cls]))
                posteriors.append(prior + conditional)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)
    
    def fit_predict(self, X, y):
        GaussianNB_Object.fit(self, X)
        return GaussianNB_Object.predict(self, y)

class MultinomialNB_Object(NaiveBayes_Object):
    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.classes = np.unique(y)
        self.likelihoods = {}
        self.prior = {}

        for cls in self.classes:
            X_cls = X[y == cls]
            self.likelihoods[cls] = (X_cls.sum(axis=0) + 1) / (X_cls.sum() + X_cls.shape[1])
            self.prior[cls] = X_cls.shape[0] / X.shape[0]

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            posteriors = []
            for cls in self.classes:
                prior = np.log(self.prior[cls])
                conditional = np.sum(x * np.log(self.likelihoods[cls]))
                posteriors.append(prior + conditional)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)
    
    def fit_predict(self, X, y):
        MultinomialNB_Object.fit(self, X)
        return MultinomialNB_Object.predict(self, y)

class BernoulliNB_Object(NaiveBayes_Object):
    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.classes = np.unique(y)
        self.likelihoods = {}
        self.prior = {}

        for cls in self.classes:
            X_cls = X[y == cls]
            self.likelihoods[cls] = (X_cls.sum(axis=0) + 1) / (X_cls.shape[0] + 2)
            self.prior[cls] = X_cls.shape[0] / X.shape[0]

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            posteriors = []
            for cls in self.classes:
                prior = np.log(self.prior[cls])
                conditional = np.sum(x * np.log(self.likelihoods[cls]) + (1 - x) * np.log(1 - self.likelihoods[cls]))
                posteriors.append(prior + conditional)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)
    
    def fit_predict(self, X, y):
        BernoulliNB_Object.fit(self, X)
        return BernoulliNB_Object.predict(self, y)