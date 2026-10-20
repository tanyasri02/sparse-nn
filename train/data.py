"""Dataset: scikit-learn's bundled 8x8 handwritten digits (1797 samples, 10 classes).
scikit-learn is used ONLY here, ONLY to load the data (no download needed)."""
import numpy as np
from sklearn.datasets import load_digits


def load(split_seed=0, test_frac=0.2):
    X, y = load_digits(return_X_y=True)
    idx = np.random.default_rng(split_seed).permutation(len(y))
    n_test = int(len(y) * test_frac)
    te, tr = idx[:n_test], idx[n_test:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8  # train-set statistics only
    f = lambda a: (a - mu) / sd
    return f(X[tr]), y[tr], f(X[te]), y[te]
