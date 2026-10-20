"""Finite-difference checks for every differentiable op in engine.tensor."""
import unittest
import numpy as np
from engine.tensor import Tensor, softmax_cross_entropy


def numgrad(f, arrays, eps=1e-6):
    grads = []
    for a in arrays:
        g = np.zeros_like(a)
        it = np.nditer(a, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = a[i]
            a[i] = old + eps; fp = f()
            a[i] = old - eps; fm = f()
            a[i] = old
            g[i] = (fp - fm) / (2 * eps)
        grads.append(g)
    return grads


def check(tc, build, *shapes, positive=False, seed=0):
    """build(*tensors) -> Tensor; compares d sum(build * R)/d inputs analytically vs numerically."""
    rng = np.random.default_rng(seed)
    arrays = [rng.uniform(0.5, 2.0, s) if positive else rng.normal(size=s) for s in shapes]
    out0 = build(*[Tensor(a) for a in arrays]).data
    R = rng.normal(size=np.shape(out0))  # random projection -> tests full Jacobian, not just column sums
    ts = [Tensor(a, requires_grad=True) for a in arrays]
    (build(*ts) * R).sum().backward()
    num = numgrad(lambda: float((build(*[Tensor(a) for a in arrays]).data * R).sum()), arrays)
    for t, n in zip(ts, num):
        np.testing.assert_allclose(t.grad, n, rtol=1e-5, atol=1e-7)
        tc.assertEqual(t.grad.shape, t.shape)


class GradCheck(unittest.TestCase):
    def test_add_broadcast(self):   check(self, lambda a, b: a + b, (4, 3), (3,))
    def test_sub_broadcast(self):   check(self, lambda a, b: a - b, (4, 3), (4, 1))
    def test_mul_broadcast(self):   check(self, lambda a, b: a * b, (4, 3), (3,))
    def test_div_broadcast(self):   check(self, lambda a, b: a / b, (4, 3), (3,), positive=True)
    def test_scalar_ops(self):      check(self, lambda a: 2.0 - a * 3.0 + 1.0 / a - (-a), (3, 2), positive=True)
    def test_matmul(self):          check(self, lambda a, b: a @ b, (4, 3), (3, 5))
    def test_sum_all(self):         check(self, lambda a: a.sum(), (3, 4))
    def test_sum_axis(self):        check(self, lambda a: a.sum(axis=0), (3, 4))
    def test_sum_keepdims(self):    check(self, lambda a: a.sum(axis=1, keepdims=True), (3, 4))
    def test_mean(self):            check(self, lambda a: a.mean(axis=0) + a.mean(), (3, 4))
    def test_tanh(self):            check(self, lambda a: a.tanh(), (3, 4))
    def test_sigmoid(self):         check(self, lambda a: a.sigmoid(), (3, 4))

    def test_relu(self):
        check(self, lambda a: a.relu(), (3, 4))  # random inputs are far from the kink at 0
        rng = np.random.default_rng(1)
        x = rng.normal(size=(5, 5)); x[np.abs(x) < 0.1] = 0.5
        t = Tensor(x, requires_grad=True); t.relu().sum().backward()
        np.testing.assert_array_equal(t.grad, (x > 0).astype(float))

    def test_softmax_ce(self):
        y = np.array([0, 2, 1, 2])
        check(self, lambda z: softmax_cross_entropy(z, y), (4, 3))

    def test_softmax_ce_stable(self):
        z = Tensor(np.array([[1000.0, -1000.0, 0.0]]), requires_grad=True)
        loss = softmax_cross_entropy(z, np.array([0]))
        loss.backward()
        self.assertTrue(np.isfinite(loss.data) and np.all(np.isfinite(z.grad)))

    def test_shared_subexpression(self):
        # x used in three places: gradients must accumulate
        check(self, lambda x: x * x + x.tanh() * x + x, (3, 3))

    def test_mlp_composite(self):
        y = np.array([1, 0, 2])
        check(self, lambda x, w1, b1, w2: softmax_cross_entropy(((x @ w1 + b1).tanh()) @ w2, y),
              (3, 4), (4, 5), (5,), (5, 3))

    def test_backward_twice_accumulates_on_leaf(self):
        x = Tensor(np.ones(3), requires_grad=True)
        (x * 2.0).sum().backward(); (x * 2.0).sum().backward()
        np.testing.assert_array_equal(x.grad, 4.0 * np.ones(3))


if __name__ == "__main__":
    unittest.main()
