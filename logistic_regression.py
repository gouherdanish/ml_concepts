import numpy as np

class LogisticRegression:
    def __init__(self,learning_rate=0.01,iterations=100) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.b0 = 0
        self.b1 = 0
        self.loss_history = []

    def _linear_model(self,z):
        return self.b0 + self.b1*z

    def _sigmoid(self,z):
        return 1/(1 + np.exp(-self._linear_model(z)))
    
    def predict_proba(self,X):
        return self._sigmoid(X)
    
    def predict(self,X):
        y_prob = self.predict_proba(X)
        return np.where(y_prob > 0.5,1,0)

    def fit(self,X,y):
        n = len(X)
        for i in range(self.iterations):
            y_pred = self.predict_proba(X)
            residual = y - y_pred

            loss = sum(-y*np.log(y_pred) - (1-y)*np.log(1-y_pred))
            self.loss_history.append(loss)

            db0 = (-2/n) * sum(residual * 1)
            db1 = (-2/n) * sum(residual * X)

            self.b0 += (-1 * self.learning_rate * db0)
            self.b1 += (-1 * self.learning_rate * db1)

    def history(self):
        return self.loss_history
    
if __name__=='__main__':
    X = np.array([1,2,-1,5,-2])
    y = np.array([0,1,0,1,0])

    X_test = np.array([3,-3,0.1,-0.1,-0.05,25,-20])

    model = LogisticRegression()
    model.fit(X,y)
    print(model.history())
    print(model.b0, model.b1)

    y_val_prob = model.predict_proba(X)
    y_val = model.predict(X)
    print(y_val_prob, y_val)

    y_test_prob = model.predict_proba(X_test)
    y_test = model.predict(X_test)
    print(y_test_prob, y_test)

