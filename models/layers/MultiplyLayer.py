import numpy as np


class MultiplyGate():
    def __init__():
        print(f'\n Initializing MultiplyGate...')
        self.X = 0
        self.Y = 0

    def forward(self, X, Y):
        self.X = X
        self.Y = Y
        return np.multiply(X, Y)

    def backward(self, dZ):
        dX = np.multiply(self.X, dZ)
        dY = np.multiply(self.Y, dZ)
        return [dX, dY]
