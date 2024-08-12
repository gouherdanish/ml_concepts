import numpy as np

class LogisticRegression:
    def __init__(self,learning_rate=0.01,iterations=100) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.b0 = 0
        self.b1 = 0

    def _bdot(self,z):
        return self.b0 + self.b1*z

    def _sigmoid(self,z):
        return 1/(1 + np.exp(-self._bdot(z)))
    
    def predict(self,X):
        return np.where(self._bdot(X) > 0,1,0)

    def fit(self,X,y):
        pass