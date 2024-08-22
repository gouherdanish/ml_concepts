import numpy as np

class LogisticRegression:
    """
    Features
        - Binary Classifier
            - 0: negative class
            - 1: positive class
        - Extended for multiple predictor variable 
    """
    def __init__(self,learning_rate=0.01,iterations=100) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        self._loss_history = []

    def __init_params__(self,n):
        """
        Initializes model parameters
        - weights and bias of the linear model are the only parameters the model will learn
        - Iniatization is assumed to be fixed and all zeros

        Args:
            n - represents number of feature variables

            _weights - vector of shape (n*1) all initialized to 0
            _bias - scalar initialized to 0
        """
        self._weights = np.zeros(n)
        self._bias = 0

    def parameters(self):
        return self._weights, self._bias
    
    def _linear_model(self,X):
        """
        Applies linear model to feature vector X

        Args:
            X - vector of shape (m*n)
        
        Returns:
            z - vector of shape (m*1)
        """
        return self._bias + np.dot(X,self._weights)

    def _sigmoid(self,X):
        """
        Calculates sigmoid of X

        Args:
            X - vector of shape (m*n)
        
        Returns:
            y_prob - represents predicted probabilities - vector of shape (m*1)
        """
        return 1/(1 + np.exp(-self._linear_model(X)))
    
    def predict_proba(self,X):
        """
        Applies sigmoid function to X

        Args:
            X: vector having shape (m*n) meaning each example having n predictor variables

        Returns:
            y: vector having shape (m*1) meaning each example having one predicted probability
        """
        return self._sigmoid(X)
    
    def predict(self,X):
        """
        Calculates Prediction for given X and applies thresholding wrt 0.5 to provide a binary outcome
        """
        y_prob = self.predict_proba(X)
        return np.where(y_prob > 0.5,1,0)

    def fit(self,X,y):
        """
        Training function which fits the features X wrt labels y
        - predicts for given X
        - finds error in prediction
        - updates model params (w and b) using gradient descent
        """
        # Number of Examples (m) and Features (n)
        m, n = X.shape

        # Initialize params
        self.__init_params__(n)

        # Repeat until converged or run for large iterations
        for i in range(self.iterations):
            # predict
            y_pred = self.predict_proba(X)

            # find error
            residual = y - y_pred

            # track loss
            loss = sum(-y*np.log(y_pred) - (1-y)*np.log(1-y_pred))
            self._loss_history.append(loss)

            # compute gradient
            db = (-2/m) * sum(residual)
            dw = (-2/m) * np.dot(residual,X)

            # update model params
            self._bias += (-1 * self.learning_rate * db)
            self._weights += (-1 * self.learning_rate * dw)

    def history(self):
        return self._loss_history
    
    
if __name__=='__main__':
    X = np.array([(-1,1),(2,1),(-1,-1),(5,2),(2,-3)])
    y = np.array([0,1,0,1,0])

    X_test = np.array([(3,1),(-3,-1),(-1,3),(-2,-2)])

    model = LogisticRegression()
    model.fit(X,y)
    print(model.history())
    print(model.parameters())

    y_val_prob = model.predict_proba(X)
    y_val = model.predict(X)
    print(y_val_prob, y_val)

    y_test_prob = model.predict_proba(X_test)
    y_test = model.predict(X_test)
    print(y_test_prob, y_test)

