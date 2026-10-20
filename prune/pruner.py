"""Importance scores, the cubic sparsity schedule, mask updates and optional regrowth."""
import numpy as np


def cubic_sparsity(step, s_init, s_final, t_start, t_end):
    """Zhu & Gupta (2017): s_t = s_f + (s_i - s_f) * (1 - progress)^3, progress in [0, 1]."""
    p = np.clip((step - t_start) / max(t_end - t_start, 1), 0.0, 1.0)
    return s_final + (s_init - s_final) * (1.0 - p) ** 3


class Pruner:
    """Layer-wise progressive pruning.

    method="magnitude": score = |w|
    method="taylor":    score = E_batch[(g_b * w)^2]  (squared first-order Taylor term, averaged
                        over the minibatches since the last pruning event; see DESIGN.md §1)
    Pruned connections stay pruned (monotone masks) unless `regrow` is called explicitly.
    """

    def __init__(self, layers, opt, method, s_final, t_start, t_end, every, s_init=0.0):
        assert method in ("magnitude", "taylor")
        self.layers, self.opt, self.method = layers, opt, method
        self.s_init, self.s_final, self.t_start, self.t_end, self.every = s_init, s_final, t_start, t_end, every
        self.acc = [np.zeros_like(l.W.data) for l in layers]
        self.n_acc = 0

    def observe(self):
        """Call after backward(), before opt.step(): accumulate Taylor statistics."""
        if self.method == "taylor":
            for a, l in zip(self.acc, self.layers):
                a += (l.W.grad * l.W.data) ** 2
            self.n_acc += 1

    def scores(self, i):
        l = self.layers[i]
        s = np.abs(l.W.data) if self.method == "magnitude" else self.acc[i] / max(self.n_acc, 1)
        return np.where(l.W.mask > 0, s, -np.inf)  # inactive always ranked lowest

    def maybe_prune(self, step):
        if step < self.t_start or step > self.t_end or (step - self.t_start) % self.every:
            return False
        self.prune_to(cubic_sparsity(step, self.s_init, self.s_final, self.t_start, self.t_end))
        return True

    def prune_to(self, sparsity):
        for i, l in enumerate(self.layers):
            n = l.W.data.size
            k_keep = n - int(round(sparsity * n))
            order = np.argsort(self.scores(i), axis=None, kind="stable")[::-1]
            new = np.zeros(n)
            new[order[:k_keep]] = 1.0
            new = new.reshape(l.W.shape)
            dropped = (l.W.mask > 0) & (new == 0)
            l.W.mask = new
            l.W.data *= new
            self.opt.reset_state(l.W, dropped)
            self.acc[i][:] = 0.0
        self.n_acc = 0


def regrow(layer, opt, k):
    """RigL-style: reactivate the k inactive weights with largest dense gradient |dL/dW_eff|.
    Reactivated weights start at 0 with fresh optimizer state (m = v = t = 0)."""
    g = np.where(layer.W.mask > 0, -np.inf, np.abs(layer.W_eff.grad))
    idx = np.unravel_index(np.argsort(g, axis=None)[::-1][:k], g.shape)
    grow = np.zeros_like(layer.W.mask, dtype=bool)
    grow[idx] = True
    grow &= layer.W.mask == 0
    layer.W.mask = layer.W.mask.copy()
    layer.W.mask[grow] = 1.0
    layer.W.data[grow] = 0.0
    opt.reset_state(layer.W, grow)
    return grow


def sparsity(layers):
    tot = sum(l.W.mask.size for l in layers)
    return 1.0 - sum(l.W.mask.sum() for l in layers) / tot
