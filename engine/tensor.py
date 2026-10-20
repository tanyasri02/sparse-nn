"""Minimal reverse-mode automatic differentiation on NumPy arrays."""
import numpy as np


def _unbroadcast(g, shape):
    """Sum a gradient back down to `shape` (reverses NumPy broadcasting)."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


def _wrap(x):
    return x if isinstance(x, Tensor) else Tensor(x)


class Tensor:
    """Array + gradient + graph edges (parents and the closure that produced it)."""

    def __init__(self, data, requires_grad=False, parents=(), op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._parents = parents
        self._op = op
        self._backward = None  # g_out -> tuple of grads, one per parent

    shape = property(lambda self: self.data.shape)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._op!r})"

    def _make(self, data, parents, op, backward):
        out = Tensor(data, any(p.requires_grad for p in parents), parents, op)
        if out.requires_grad:
            out._backward = backward
        return out

    # ---- element-wise arithmetic (broadcasting) ----
    def __add__(self, o):
        o = _wrap(o)
        return self._make(self.data + o.data, (self, o), "add",
                          lambda g: (_unbroadcast(g, self.shape), _unbroadcast(g, o.shape)))

    def __sub__(self, o):
        o = _wrap(o)
        return self._make(self.data - o.data, (self, o), "sub",
                          lambda g: (_unbroadcast(g, self.shape), _unbroadcast(-g, o.shape)))

    def __mul__(self, o):
        o = _wrap(o)
        return self._make(self.data * o.data, (self, o), "mul",
                          lambda g: (_unbroadcast(g * o.data, self.shape),
                                     _unbroadcast(g * self.data, o.shape)))

    def __truediv__(self, o):
        o = _wrap(o)
        return self._make(self.data / o.data, (self, o), "div",
                          lambda g: (_unbroadcast(g / o.data, self.shape),
                                     _unbroadcast(-g * self.data / o.data ** 2, o.shape)))

    def __neg__(self):
        return self * -1.0

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = lambda self, o: _wrap(o) - self
    __rtruediv__ = lambda self, o: _wrap(o) / self

    # ---- linear algebra / reductions ----
    def __matmul__(self, o):
        o = _wrap(o)
        return self._make(self.data @ o.data, (self, o), "matmul",
                          lambda g: (g @ o.data.T, self.data.T @ g))

    def sum(self, axis=None, keepdims=False):
        def bw(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return (np.broadcast_to(g, self.shape).copy(),)
        return self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum", bw)

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else np.prod([self.shape[a] for a in np.atleast_1d(axis)])
        return self.sum(axis, keepdims) / float(n)

    # ---- activations ----
    def relu(self):
        m = self.data > 0
        return self._make(self.data * m, (self,), "relu", lambda g: (g * m,))

    def tanh(self):
        y = np.tanh(self.data)
        return self._make(y, (self,), "tanh", lambda g: (g * (1 - y ** 2),))

    def sigmoid(self):
        y = 0.5 * (1 + np.tanh(0.5 * self.data))  # overflow-free form
        return self._make(y, (self,), "sigmoid", lambda g: (g * y * (1 - y),))

    # ---- backprop ----
    def backward(self, grad=None):
        topo, seen, stack = [], set(), [(self, False)]
        while stack:  # iterative post-order DFS -> topological order
            t, done = stack.pop()
            if done:
                topo.append(t)
                continue
            if id(t) in seen:
                continue
            seen.add(id(t))
            stack.append((t, True))
            stack.extend((p, False) for p in t._parents if id(p) not in seen)
        self.grad = np.ones_like(self.data) if grad is None else np.asarray(grad, dtype=np.float64)
        for t in reversed(topo):
            if t._backward is None or t.grad is None:
                continue
            for p, g in zip(t._parents, t._backward(t.grad)):
                if p.requires_grad:  # accumulate: a tensor may feed several ops
                    p.grad = g if p.grad is None else p.grad + g


def softmax_cross_entropy(logits, labels):
    """Mean cross-entropy of softmax(logits) against integer labels (log-sum-exp stable)."""
    z = logits.data - logits.data.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    n = len(labels)
    loss = -logp[np.arange(n), labels].mean()

    def bw(g):
        d = np.exp(logp)
        d[np.arange(n), labels] -= 1.0
        return (g * d / n,)
    return logits._make(loss, (logits,), "softmax_ce", bw)
