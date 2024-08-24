import numpy as np
import pandas as pd

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
        self._gradients = []

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
            residual = y_pred - y

            # track loss
            loss = (1/m) * sum(-y*np.log(y_pred) - (1-y)*np.log(1-y_pred))
            self._loss_history.append(loss)

            # # compute gradient
            dLdb = (1/m) * sum(residual)
            dLdw = (1/m) * np.dot(residual,X)
            self._gradients.append((dLdb,dLdw))

            # update model params
            self._bias += (-1 * self.learning_rate * dLdb)
            self._weights += (-1 * self.learning_rate * dLdw)

            if i == 0 or (i + 1) % 10 == 0:
                print(f"EPOCH [{i+1}/{self.iterations}]: training loss = {loss}")

    def history(self):
        return self._loss_history

    def parameters(self):
        return self._weights, self._bias

    def gradients(self):
        return self._gradients
    
    
if __name__=='__main__':
    # Sample Data 1
    # X = np.array([(-1,1),(2,1),(-1,-1),(5,2),(2,-3)])
    # y = np.array([0,1,0,1,0])
    # X_test = np.array([(3,1),(-3,-1),(-1,3),(-2,-2)])

    """
    Sample Data 2

    Let's consider an example for a Binary Classification Problem

    "_Gouher browses movies through OTT and his decision to watch a particular movie is based on its rating and release date._"

    Let's take some historical data from Gouher's OTT watchlist history

    | movie name    | rating | released date | watched | 
    | ------------- | ------ | ------------- | ------- |
    | kalki         |   6.2  |     2024      |    1    |
    | tumbbad       |   7.8  |     2018      |    1    | 
    | indiana jones |   8.1  |     1990      |    0    | 
    | Tiger 3       |   4.5  |     2023      |    0    | 

    """
    # Data Gathering
    df = pd.DataFrame({
        'movie':['Kalki','Tumbbad','Indiana Jones','Tiger 3'],
        'rating':[6.2,7.8,8.1,4.5],
        'released_date':[2024,2018,1990,2023],
        'watched':[1,1,0,0]
    })

    # Feature Engg
    df['recency'] = pd.Timestamp.today().year - df['released_date']

    # Training Data Prep
    X = df[['rating','recency']].to_numpy()
    y = df['watched'].to_numpy()

    # Model Training
    model = LogisticRegression(iterations=100,learning_rate=0.2)
    model.fit(X,y)
    print(model.parameters())

    # Validating
    y_val_prob = model.predict_proba(X)
    y_val = model.predict(X)
    print(y_val_prob, y_val)

    # Validation Accuracy
    val_acc = 100*sum(y_val==y)/len(y)
    print(f'validation accuracy = {val_acc}%')

    # Testing
    X_test = np.array([8.9,2])  # Checking for The Batman movie
    y_test_prob = model.predict_proba(X_test)
    y_test = model.predict(X_test)
    print(y_test_prob, y_test)


