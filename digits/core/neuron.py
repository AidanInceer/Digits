import random

from digits.core.value import Value


class Neuron:

    def __init__(self, nin) -> None:
        self.weight = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):

        activation = sum((wi * xi for wi, xi in zip(self.weight, x)), self.bias)
        out = activation.tanh()
        return out

    def parameters(self):
        return self.weight + [self.bias]
