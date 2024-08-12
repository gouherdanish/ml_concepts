import numpy as np

class LogisticRegression:
    def __init__(self,learning_rate=0.01,iterations=100) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.b0 = 0
        self.b1 = 0
        self.loss_history = []

    def _bdot(self,z):
        return self.b0 + self.b1*z

    def _sigmoid(self,z):
        return 1/(1 + np.exp(-self._bdot(z)))
    
    def predict(self,X):
        return self._sigmoid()
        # return np.where(self._bdot(X) > 0,1,0)

    def fit(self,X,y):
        n = len(X)
        y_pred = self.predict(X)
        residual = y - y_pred
        loss = (1.0/n)*sum(residual**2)
        self.loss_history.append(loss)

        db0 = (-2/n) * residual * 1
        db1 = (-2/n) * residual * X

        self.b0 += (-1 * self.learning_rate * db0)
        self.b1 += (-1 * self.learning_rate * db1)

    def history(self):
        return self.loss_history