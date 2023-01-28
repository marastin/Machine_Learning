"Last Update: 1/28/2023"

import numpy as np

class Percepteron:

    def __init__(self, lr=0.01, n_iter=50, random_seed=1):
        self.lr = lr
        self.n_iter = n_iter
        self.random_seed = random_seed
    

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_seed)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.error_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.error_.append(errors)
        
        return self
    

    def forward(self, X):
        return np.dot(X, self.w_) + self.b_
    

    def predict(self, X):
        return np.where(self.forward(X) > 0.0 , 1, 0 )


if __name__ == "__main__":
    print("Percepteron Classifier")