import numpy as np
from custom_functions import sigmoid


class RL():

    def __init__(self, threshold = 0.5):

        self.threshold = threshold
        self.TH = None  # poids, créés dans fit() quand on connaît le nombre de colonnes de X


    def cost_calc(self, X, y):

        y_pred = sigmoid(np.dot(X, self.TH))

        # Évite log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def grad_calc(self, X, y): #dCOST_dTH

        N = X.shape[0]

        y_pred = sigmoid(np.dot(X, self.TH))

        Xt = X.T

        grad = 1 / N * np.dot(Xt, y_pred - y)

        return grad

    def fit(self, X, y, learning_rate = 0.01, n_iterations = 1000):

        X = np.asarray(X, dtype = float)
        y = np.asarray(y, dtype = float).reshape(-1, 1)

        self.TH = np.random.rand(X.shape[1], 1)

        for step in range(n_iterations):

            self.TH = self.TH - learning_rate * self.grad_calc(X, y)

        self.training_cost = self.cost_calc(X, y)

        return self


    def predict(self, X):

        prediction = sigmoid(np.dot(np.asarray(X, dtype = float), self.TH))

        return (prediction > self.threshold).astype(int).ravel()
