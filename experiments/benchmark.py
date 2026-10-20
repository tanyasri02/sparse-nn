"""Inference cost: dense baseline vs pruned models (Taylor, seed 0).

    python -m experiments.benchmark   -> experiments/results/efficiency.{json,png}

Paths timed (all pure NumPy, single BLAS thread, median of repeats):
  dense       : x @ W + b with the full matrices (what a masked model costs if run densely)
  compact     : dense BLAS on matrices with dead units removed (exact, see prune/sparse_infer.py)
  sparse_csc  : gather kernel over non-zeros only, O(batch * nnz)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train.trainer import run
from train.data import load
from prune.sparse_infer import SparseMLP, CompactDenseMLP

OUT = os.path.join(os.path.dirname(__file__), "results")
TARGETS = [0.0, 0.5, 0.75, 0.9, 0.95, 0.98]
BATCHES = [1, 64, 1024]


def dense_fn(model):
    blocks = [(l.W.data * l.W.mask, l.b.data) for l in model.linears()]
    def f(x):
        for i, (W, b) in enumerate(blocks):
            x = x @ W + b
            if i < len(blocks) - 1:
                x = np.maximum(x, 0.0)
        return x
    return f


def latency_us(f, x, budget=0.4):
    f(x); n = 1
    while True:  # calibrate loop count, then take median of 7 repeats
        t = time.perf_counter(); [f(x) for _ in range(n)]; dt = time.perf_counter() - t
        if dt > budget / 7: break
        n *= 2
    reps = []
    for _ in range(7):
        t = time.perf_counter(); [f(x) for _ in range(n)]; reps.append((time.perf_counter() - t) / n)
    return float(np.median(reps) * 1e6)


def main():
    Xte = load()[2]
    rng = np.random.default_rng(0)
    rows = []
    for t in TARGETS:
        model, h = run(seed=0, method="taylor" if t else None, target=t)
        lin = model.linears()
        total_w = sum(l.W.data.size for l in lin); total_b = sum(l.b.data.size for l in lin)
        active_w = int(sum(l.W.mask.sum() for l in lin))
        sp, cd = SparseMLP(model), CompactDenseMLP(model)
        x0 = Xte[:64]
        assert np.allclose(sp(x0), dense_fn(model)(x0)) and np.allclose(cd(x0), dense_fn(model)(x0))
        r = dict(target=t, sparsity=h[-1]["sparsity"], test_acc=h[-1]["test_acc"],
                 params_total=total_w + total_b, params_active=active_w + total_b,
                 macs_dense=total_w, macs_sparse=sp.macs(), macs_compact=cd.macs(),
                 hidden_units=[W.shape[1] for W, _, _ in cd.blocks[:-1]],
                 bytes_dense=8 * (total_w + total_b), bytes_sparse=sp.nbytes(), latency_us={})
        for B in BATCHES:
            x = Xte[rng.integers(0, len(Xte), B)]
            r["latency_us"][B] = {k: latency_us(f, x) for k, f in
                                  (("dense", dense_fn(model)), ("compact", cd), ("sparse_csc", sp))}
        rows.append(r)
        print(json.dumps(r))
    with open(f"{OUT}/efficiency.json", "w") as f:
        json.dump(rows, f, indent=2)

    fig, ax = plt.subplots(1, len(BATCHES), figsize=(13, 3.8))
    xs = [r["sparsity"] * 100 for r in rows]
    for a, B in zip(ax, BATCHES):
        for k, c in (("dense", "#555"), ("compact", "#2ca02c"), ("sparse_csc", "#d62728")):
            a.plot(xs, [r["latency_us"][B][k] for r in rows], marker="o", color=c, label=k)
        a.set_title(f"batch={B}"); a.set_xlabel("sparsity (%)"); a.set_ylabel("latency (µs, median)")
        a.set_yscale("log"); a.grid(alpha=.3); a.legend()
    fig.suptitle("Measured inference latency (NumPy, 1 thread)"); fig.tight_layout()
    fig.savefig(f"{OUT}/latency.png", dpi=150)


if __name__ == "__main__":
    main()
