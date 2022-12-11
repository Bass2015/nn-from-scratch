import numpy as np
import math

class FullyConected:
    def __init__(self, ins, outs, activation='relu') -> None:
        self.weights = np.random.normal(0, 1/(ins**.5), (ins, outs))
        self.bias = np.random.normal(0, 1/(ins**.5), outs)
        self.activation = activation

    def forward(self, input):
        self.input = input
        self.z = np.matmul(input, self.weights) + self.bias
        if self.activation == 'relu':
            self.output =  ReLU(self.z)
        else: 
            self.output = self.z
        return self.output
    
    def backward(self, dy, lr, batch_size):
        if self.activation == 'relu':
            dy *= np.greater(self.z, 0)
        dx = np.dot(dy, self.weights.T)
        self.dw = np.dot(self.input.T, dy) / batch_size
        self.db = dy.mean(axis=0)
        return dx

def ReLU(x):
    return np.maximum(0, x)
