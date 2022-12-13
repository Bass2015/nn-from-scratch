import numpy as np

class FullyConected:
    def __init__(self, ins, outs, activation='relu'):
        self.weights = np.random.normal(0, 1/(ins**.5), (ins, outs))
        self.bias = np.random.normal(0, 1/(ins**.5), outs)
        self.activation = activation

    def forward(self, input):
        """Saves the input for the backward pass. Calculates the weighted sum, 
        applies ReLU if needed, saves the output and returns it. 
        Performs matrix multiplications, so the inputs must come in form of matrix.
        
        Parameters
        ------------
        input: numpy.array
            Array of shape (batch_size, num_inputs)]    
        """
        self.input = input
        self.z = np.matmul(input, self.weights) + self.bias
        if self.activation == 'relu':
            self.output =  ReLU(self.z)
        else: 
            self.output = self.z
        return self.output
    
    def backward(self, dy, batch_size):
        if self.activation == 'relu':
            dy *= np.greater(self.z, 0)
        dx = np.dot(dy, self.weights.T)
        self.dw = np.dot(self.input.T, dy) / batch_size
        self.db = dy.mean(axis=0)
        return dx

def ReLU(x):
    return np.maximum(0, x)
