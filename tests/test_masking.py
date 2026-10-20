"""Masking semantics: forward, backward, Adam state, prune -> update -> regrow regression."""
import unittest
import numpy as np
from engine.tensor import Tensor, softmax_cross_entropy
from nn.layers import mlp
from optim.adam import Adam
from prune.pruner import Pruner, regrow, sparsity


def setup(seed=0):
    rng = np.random.default_rng(seed)
    model = mlp([6, 8, 3], rng)
    x, y = rng.normal(size=(16, 6)), rng.integers(0, 3, 16)
    opt = Adam(model.parameters(), lr=1e-2)
    return model, opt, x, y


def train_step(model, opt, x, y, pruner=None):
    opt.zero_grad()
    loss = softmax_cross_entropy(model(Tensor(x)), y)
    loss.backward()
    if pruner:
        pruner.observe()
    opt.step()
    return float(loss.data)


class Masking(unittest.TestCase):
    def test_masked_weight_has_no_forward_effect(self):
        model, _, x, _ = setup()
        l = model.linears()[0]
        l.W.mask[2, 3] = 0.0
        y1 = model(Tensor(x)).data
        l.W.data[2, 3] = 1e6  # even a huge stale value must not leak through
        np.testing.assert_array_equal(model(Tensor(x)).data, y1)

    def test_gradient_semantics(self):
        model, _, x, y = setup()
        l = model.linears()[0]
        l.W.mask[2, 3] = 0.0
        softmax_cross_entropy(model(Tensor(x)), y).backward()
        self.assertEqual(l.W.grad[2, 3], 0.0)          # parameter gradient is zero
        self.assertNotEqual(l.W_eff.grad[2, 3], 0.0)   # dense signal kept for regrowth
        np.testing.assert_array_equal(l.W.grad, l.W_eff.grad * l.W.mask)

    def test_masked_weights_stay_exactly_zero_under_adam(self):
        model, opt, x, y = setup()
        pr = Pruner(model.linears(), opt, "magnitude", 0.5, 0, 0, 1)
        for _ in range(5):
            train_step(model, opt, x, y)
        pr.prune_to(0.5)
        for _ in range(20):
            train_step(model, opt, x, y)
        for l, (m, v, t) in zip(model.linears(), [(opt.m[i], opt.v[i], opt.t[i]) for i in (0, 2)]):
            off = l.W.mask == 0
            self.assertTrue(off.any())
            self.assertTrue(np.all(l.W.data[off] == 0) and np.all(m[off] == 0)
                            and np.all(v[off] == 0) and np.all(t[off] == 0))
        self.assertAlmostEqual(sparsity(model.linears()), 0.5, places=6)

    def test_prune_update_regrow_regression(self):
        """Full sequence: train -> prune -> train -> regrow -> one step.
        A reactivated weight must take a *fresh* Adam step: with m=v=t=0, the first update is
        -lr * g / (|g| + eps), i.e. ~lr in magnitude and opposite to its gradient sign."""
        model, opt, x, y = setup(3)
        l = model.linears()[0]
        for _ in range(10):
            train_step(model, opt, x, y)
        Pruner(model.linears(), opt, "magnitude", 0.75, 0, 0, 1).prune_to(0.75)
        for _ in range(10):
            train_step(model, opt, x, y)
        train_step(model, opt, x, y)  # populate W_eff.grad at current weights
        grown = regrow(l, opt, 4)
        self.assertEqual(grown.sum(), 4)
        self.assertTrue(np.all(l.W.data[grown] == 0))

        opt.zero_grad()
        softmax_cross_entropy(model(Tensor(x)), y).backward()
        g = l.W.grad[grown].copy()
        before = l.W.data.copy()
        opt.step()
        expected = -opt.lr * g / (np.abs(g) + opt.eps)
        np.testing.assert_allclose(l.W.data[grown] - before[grown], expected, rtol=1e-6)
        self.assertTrue(np.all(opt.t[0][grown] == 1))
        self.assertTrue(np.all(l.W.data[l.W.mask == 0] == 0))


if __name__ == "__main__":
    unittest.main()
