"""Adam with mask-aware, per-element state.

For each element i with mask a_i in {0,1}:
    t_i <- t_i + a_i
    m_i <- a_i * (b1 m_i + (1-b1) g_i)
    v_i <- a_i * (b2 v_i + (1-b2) g_i^2)
    w_i <- a_i * (w_i - lr * m_hat_i / (sqrt(v_hat_i) + eps)),  m_hat_i = m_i / (1 - b1^t_i), etc.
The step counter is per element, so a reactivated weight restarts with correct bias correction.
"""
import numpy as np


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = list(params), lr, b1, b2, eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = [np.zeros(p.shape, dtype=np.int64) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        for p, m, v, t in zip(self.params, self.m, self.v, self.t):
            if p.grad is None:
                continue
            a = getattr(p, "mask", np.ones_like(p.data)) > 0
            g = np.where(a, p.grad, 0.0)
            t += a
            m[...] = np.where(a, self.b1 * m + (1 - self.b1) * g, 0.0)
            v[...] = np.where(a, self.b2 * v + (1 - self.b2) * g * g, 0.0)
            tt = np.maximum(t, 1)
            upd = (m / (1 - self.b1 ** tt)) / (np.sqrt(v / (1 - self.b2 ** tt)) + self.eps)
            p.data[...] = np.where(a, p.data - self.lr * upd, 0.0)

    def reset_state(self, p, where):
        """Zero m, v, t for entries `where` (bool array) of parameter p."""
        i = next(k for k, q in enumerate(self.params) if q is p)
        self.m[i][where] = 0.0
        self.v[i][where] = 0.0
        self.t[i][where] = 0
