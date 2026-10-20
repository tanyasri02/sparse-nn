"""CLI:  python -m train.run [--method magnitude|taylor --target 0.9] [--seed 0] [--epochs 40]"""
import argparse
from train.trainer import run

p = argparse.ArgumentParser()
p.add_argument("--method", choices=["magnitude", "taylor"])
p.add_argument("--target", type=float, default=0.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=40)
a = p.parse_args()
_, hist = run(seed=a.seed, method=a.method, target=a.target, epochs=a.epochs, verbose=True)
print(f"final test_acc={hist[-1]['test_acc']:.4f} sparsity={hist[-1]['sparsity']:.4f}")
