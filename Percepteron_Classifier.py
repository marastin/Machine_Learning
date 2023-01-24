"Date: 1/24/2023"

import numpy as np

class Percepteron:

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.error_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.error_.append(errors)
        
        return self
    

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    

    def predict(self, X):
        return np.where(self.net_input(X) > 0.0 , 1, 0 )


if __name__ == "__main__":
    print("Percepteron Classifier")