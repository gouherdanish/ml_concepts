import numpy as np

class LinearRegression:
    """
    Simulates Linear Regression model
    using Gradient Descent optimization
    """
    def __init__(self,learning_rate=0.1,iterations=100) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss_history = []
        self.b0 = 0
        self.b1 = 0
    
    def predict(self,X):
        """
        Implements Hypothesis Function for Linear Model
        h = b0 + b1*X
        """
        return self.b0 + self.b1 * X
    
    def fit(self,X,y):
        """
        Implements training the model on the supplied data
        - Find y_pred using model
        - Find residual/error in prediction
        - Compute gradient wrt model params
        - Update model params in the direction of decreasing gradient
        """
        # number of examples
        n = len(X)
        # Iterate finite number of times
        for i in range(self.iterations):
            # Predict for all examples
            y_pred = self.predict(X)
            residual = y - y_pred
            loss = (1.0/n)*sum(residual**2)
            self.loss_history.append(loss)
            # Compute Gradients
            db0 = (-2/n) * sum(residual)
            db1 = (-2/n) * sum(residual * X)
            # Update params/Gradient Descent 
            self.b0 += (-1 * self.learning_rate * db0)
            self.b1 += (-1 * self.learning_rate * db1)

    def history(self):
        return self.loss_history
    

if __name__=='__main__':
    # Sample Training Data to simulate y = x
    X = np.array([1,2,3,4])
    y = np.array([1,1.9,3.1,4])

    # Sample Test Data
    X_test = np.array([5,10])
    
    # Training Step
    model = LinearRegression()
    model.fit(X,y)

    # Inference Step
    print(model.predict(X_test))
    
    # Interpreting Model Params
    print(model.history())
    print(model.b0,model.b1)