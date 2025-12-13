import numpy as np
from michigrad.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class relu(Module):
    def __call__(self, x):
        if isinstance(x, list):
            return [xi.relu() for xi in x]
        return x.relu()

class tanh(Module):
    def __call__(self, x):
        if isinstance(x, list):
            return [xi.tanh() for xi in x]
        return x.tanh()
    
class sigmoide(Module):
    def __call__(self, x):
        if isinstance(x, list):
            return [xi.sigmoide() for xi in x]
        return x.sigmoide()

class Neuron(Module):

    def __init__(self, nin):
        self.w = [Value(np.random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(np.random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"LinearNeuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts, *metodos):
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            self.layers.append(Layer(sz[i], sz[i+1]))
            if i < len(metodos):
                self.layers.append(metodos[i])

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        if isinstance(x, list) and len(x) == 1:
            return x[0]
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"