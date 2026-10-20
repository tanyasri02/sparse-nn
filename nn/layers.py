"""Composable layers built on engine.Tensor."""
import numpy as np
from engine.tensor import Tensor


class Module:
    def parameters(self):
        return []


class Linear(Module):
    """y = x @ (W * mask) + b.  He-normal init, zero bias.

    `W.mask` is a {0,1} array owned by the weight. The forward pass multiplies by it, so a
    masked entry contributes exactly zero and receives dL/dW = mask * G = 0, while the dense
    signal G = dL/dW_eff is kept on `self.W_eff.grad` for regrowth decisions.
    """

    def __init__(self, fan_in, fan_out, rng):
        self.W = Tensor(rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out)), requires_grad=True)
        self.W.mask = np.ones((fan_in, fan_out))
        self.b = Tensor(np.zeros(fan_out), requires_grad=True)
        self.W_eff = None

    def __call__(self, x):
        self.W_eff = self.W * self.W.mask
        return x @ self.W_eff + self.b

    def parameters(self):
        return [self.W, self.b]


class ReLU(Module):
    np_fn = staticmethod(lambda x: np.maximum(x, 0.0))
    def __call__(self, x): return x.relu()


class Tanh(Module):
    np_fn = staticmethod(np.tanh)
    def __call__(self, x): return x.tanh()


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def linears(self):
        return [l for l in self.layers if isinstance(l, Linear)]


def mlp(sizes, rng, act=ReLU):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(Linear(a, b, rng))
        if i < len(sizes) - 2:
            layers.append(act())
    return Sequential(*layers)
