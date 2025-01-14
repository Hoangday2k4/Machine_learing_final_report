import numpy as np

class Logistic_Regression_Object:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        self.m, self.n = X.shape
        self.theta = np.zeros(self.n)
        self.bias = 0
        self.X = X
        self.y = y
        
        for _ in range(self.iterations):
            model = np.dot(X, self.theta) + self.bias
            y_pred = self.sigmoid(model)
            
            d_theta = (1 / self.m) * np.dot(X.T, (y_pred - y))
            d_bias = (1 / self.m) * np.sum(y_pred - y)
            
            self.theta -= self.learning_rate * d_theta
            self.bias -= self.learning_rate * d_bias
    
    def predict(self, X):
        model = np.dot(X, self.theta) + self.bias
        y_pred = self.sigmoid(model)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(y_pred_class)