"""End-to-end: reproducibility, stability, pruning target, exact sparse inference."""
import unittest
import numpy as np
from train.trainer import run
from train.data import load
from engine.tensor import Tensor
from prune.pruner import cubic_sparsity
from prune.sparse_infer import SparseMLP, CompactDenseMLP


class Training(unittest.TestCase):
    def test_reproducible_and_learns(self):
        _, h1 = run(seed=1, epochs=3)
        _, h2 = run(seed=1, epochs=3)
        self.assertEqual(h1, h2)
        self.assertLess(h1[-1]["train_loss"], h1[0]["train_loss"])
        self.assertGreater(h1[-1]["test_acc"], 0.9)

    def test_schedule(self):
        s = [cubic_sparsity(t, 0.0, 0.9, 10, 110) for t in range(0, 130)]
        self.assertEqual(s[0], 0.0); self.assertAlmostEqual(s[-1], 0.9)
        self.assertTrue(all(b >= a for a, b in zip(s, s[1:])))

    def test_reaches_target_and_sparse_inference_is_exact(self):
        for method in ("magnitude", "taylor"):
            model, h = run(seed=0, epochs=6, method=method, target=0.9, prune_start=1, prune_end=4)
            self.assertAlmostEqual(h[-1]["sparsity"], 0.9, places=3)
            X = load()[2]
            dense = model(Tensor(X)).data
            np.testing.assert_allclose(SparseMLP(model)(X), dense, atol=1e-10)
            np.testing.assert_allclose(CompactDenseMLP(model)(X), dense, atol=1e-10)

    def test_compaction_folds_dead_unit_bias(self):
        model, _ = run(seed=0, epochs=1)
        l1 = model.linears()[0]
        l1.W.mask[:, :5] = 0.0; l1.b.data[:5] = 0.7  # 5 units with constant output relu(0.7)
        X = load()[2]
        np.testing.assert_allclose(SparseMLP(model)(X), model(Tensor(X)).data, atol=1e-10)
        self.assertEqual(SparseMLP(model).blocks[0][0].fan_out, 256 - 5)


if __name__ == "__main__":
    unittest.main()
