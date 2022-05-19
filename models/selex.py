import numpy as np
from utils.math import *

# numpy configuration
rng = np.random.default_rng(4)
np.set_printoptions(precision=5)


class SELEX_Net():
    def __init__(self, alpha, lamb, epochs, params):
        m, n, l1 = params[0], params[1], params[2]
        print(f'\n Initializing new fully connected model...')
        self.alpha = alpha
        self.lamb = lamb
        self.epochs = epochs
        self.params = {
            "W1": rng.standard_normal([n, l1]) * 0.1,
            "b1": np.zeros([1, l1]),
            "W2": rng.standard_normal([l1, 1]) * 0.1,
            "b2": np.zeros([1, 1])
        }
        print(f'  W1: {self.params["W1"].shape}  -  W2: {self.params["W2"].shape}')

    def train(self, X, Y, m, n, prnt):
        print(f'\n  Beginning Gradient Descent - {self.epochs} epochs')
        hist = np.zeros([3, self.epochs])
        e = 0
        while e < self.epochs:
            if (e%prnt==0): print(f"\n  Starting Epoch: {e}")
            Z1, A1, Z2, A2 = self.compute_activations(X)
            dEW2, dEW1 = self.compute_gradients(Y, X, Z1, A1, Z2, A2)
            self.update_params(dEW2, dEW1, m)
            cost, acc, err = self.test(Y, A2)
            hist[0, e], hist[1,e], hist[2,e] = cost, acc, err
            if (e%prnt==0): print(f"   cost: {np.round(hist[0,e],4)}  acc: {np.round(hist[1,e],4)}%")
            e += 1
        return hist

    def compute_activations(self, A0):
        Z1 = linear(A0, self.params["W1"], self.params["b1"])
        A1 = sigmoid(Z1)
        Z2 = linear(A1, self.params["W2"], self.params["b2"])
        A2 = sigmoid(Z2)
        return Z1, A1, Z2, A2

    def compute_gradients(self, Y, A0, Z1, A1, Z2, A2):
        dEA2 = -(Y-A2)
        dA2Z2 = dSigmoid(A2)
        dZ2W2 = A1
        dZ2A1 = self.params["W2"]
        dEZ2 = np.multiply(dEA2, dA2Z2)
        dEW2 = dZ2W2.T.dot(dEZ2)
        dEA1 = dEZ2.dot(dZ2A1.T)
        dA1Z1 = dSigmoid(A1)
        dEZ1 =  np.multiply(dEA1, dA1Z1)
        dZ1W1 = A0
        dZ1A0 = self.params["W1"]
        dEW1 = dZ1W1.T.dot(dEZ1)
        return dEW2, dEW1

    def update_params(self, dw2, dw1, m):
        reg = (np.sum(np.square(self.params["W1"])) + np.sum(np.square(self.params["W2"]))) * self.lamb / (2*m)
        self.params["W1"] -= dw1 * self.alpha + reg * self.lamb
        self.params["W2"] -= dw2 * self.alpha + reg * self.lamb

    def test(self, Y, A2, b):
        cost = self.cost(Y, A2)
        acc = self.accuracy(Y, A2)
        err = self.error(Y, A2)
        return cost, acc, err

    def error(self, Y, A2):
        return np.sum(np.square(Y-A2)/2)

    def cost(self, Y, A2):
        lyh = np.log(A2+0.001)
        ylyh = np.multiply(Y, lyh)
        lnyh = np.log(1-A2+0.001)
        nylnyh = np.multiply((1-Y), lnyh)
        cost = np.average(ylyh + nylnyh) * -1
        return cost

    def accuracy(self, Y, A2):
        A2[A2<0.5] = 0
        A2[A2>0.5] = 1
        eq = Y[Y==A2]
        m = Y.shape[0]
        correct = np.sum(eq)
        acc = correct/m * 100
        return acc
