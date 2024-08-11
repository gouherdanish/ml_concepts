import numpy as np

class LinearRegression:
    def __init__(self,iterations=100) -> None:
        self.iterations = iterations
        self.b0 = 0
        self.b1 = 0
    
    def predict(self,X):
        return self.b1 * X + self.b0
    
    def fit(self,X,y):
        n = len(X)                  # number of examples
        for i in self.iterations:
            y_pred = self.predict(X)
            error = y - y_pred
            d1 = (-2/n) * error