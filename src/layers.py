import numpy as np

class FullyConected:
    def __init__(self, ins, outs) -> None:
        self.weights = np.arange(ins * outs).reshape(ins, outs)
        self.bias = np.arange(outs)

    def forward(self, input):
        self.input = input
        self.z = np.matmul()
    
    def backward(self, dy):
        pass

def ReLU(x):
    return np.max(0, x)