
class Optimizer:
    def __init__(self, net, lr):
        self.net = net
        self.lr = lr

    def step(self):
        for layer in self.net.layers:
            self.update_params(layer)
    
    def update_params(layer):
        pass

class SGD(Optimizer):
    def update_params(self, layer):
        layer.weights -= self.lr * layer.dw
        layer.bias -= self.lr * layer.db