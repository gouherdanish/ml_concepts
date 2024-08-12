class LogisticRegression:
    def __init__(self,learning_rate=0.01,iterations=100) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.b0 = 0
        self.b1 = 0

    