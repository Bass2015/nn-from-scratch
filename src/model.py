
from src.layers import FullyConected as FC

class Net:
    def __init__(self, neurons, last_relu=False):
        """Parameters
           -----------
            neurons: list or tuple
                Number of neurons in each layer. 
                neurons[0] is input layer.
        """
        self.layers = []
        for n in range(len(neurons)-1):
            ins = neurons[n]
            outs = neurons[n+1]
            if n == len(neurons)-2 and not last_relu:
                activation = 'none'
            else: 
                activation = 'relu'
            layer = FC(ins, outs, activation=activation)
            self.layers.append(layer)
        
    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, dl, batch_size):
        for layer in reversed(self.layers):
            dl = layer.backward(dl, batch_size)