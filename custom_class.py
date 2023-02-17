import numpy as np
from custom_functions import sigmoid

#USE NP MEAN

class RL():

    def __init__(self):
        
        self.threshold = 0,5
        self.TH = np.random.rand(X.shape[1], 1)
        

    def cost_calc(self):

        N = self.X.shape[1]
        
        #USE np.mean

        cost = 0

        for i in range(N):
            z = np.dot(X[i], self.TH)
            
            y_pred = sigmoid(z)
            cost = cost + (y[i] * np.log(y_pred) + (1 - y[i]) * np.log(1 - y_pred))
            
        return -1 / N * cost
    
    def grad_calc(self, X, y): #dCOST_dTH

        N = X.shape[1]

        y_pred = sigmoid(np.dot(X, self.TH))       

        Xt = X.T

        grad = 1 / N * np.dot(Xt, y_pred - y)

        return grad
    
    def fit(self, X, y, learning_rate = 0.01):

        N_iterations = X.shape[0]
        start = 0
        end = 0
        
        for step in range(N_iterations):

            self.TH = self.TH - learning_rate * self.grad_calc(X, y)

        training_cost = self.cost_calc(X[:,:], y)

        
    def predict(self, X):

        prediction = sigmoid(X * self.TH)

        if prediction > self.threshold:
            
            return 1

        else:

            return 0