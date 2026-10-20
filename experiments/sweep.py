"""Sparsity-accuracy sweep: {magnitude, taylor} x targets x seeds, plus dense baseline.

    python -m experiments.sweep            # 10 seeds (default), writes experiments/results/
    python -m experiments.sweep --seeds 3  # quicker
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, csv, json
from multiprocessing import Pool
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train.trainer import run

OUT = os.path.join(os.path.dirname(__file__), "results")
TARGETS = [0.5, 0.75, 0.9, 0.95, 0.98, 0.99]
T95 = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
       14: 2.145, 19: 2.093}  # two-sided 95% Student-t critical values


def job(args):
    method, target, seed = args
    _, h = run(seed=seed, method=method, target=target)
    return dict(method=method or "dense", target=target, seed=seed, sparsity=h[-1]["sparsity"],
                test_acc=h[-1]["test_acc"], test_loss=h[-1]["test_loss"], train_acc=h[-1]["train_acc"],
                curve=[(e["epoch"], e["train_loss"], e["test_acc"], e["sparsity"]) for e in h])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10); a = ap.parse_args()
    seeds = range(a.seeds)
    jobs = [(None, 0.0, s) for s in seeds] + [(m, t, s) for m in ("magnitude", "taylor") for t in TARGETS for s in seeds]
    with Pool() as p:
        rows = p.map(job, jobs)
    os.makedirs(OUT, exist_ok=True)
    keys = ["method", "target", "seed", "sparsity", "test_acc", "test_loss", "train_acc"]
    with open(f"{OUT}/sweep_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); [w.writerow({k: r[k] for k in keys}) for r in rows]
    with open(f"{OUT}/curves.json", "w") as f:
        json.dump([{k: r[k] for k in ("method", "target", "seed", "curve")} for r in rows], f)

    acc = lambda m, t: np.array([r["test_acc"] for r in sorted(rows, key=lambda r: r["seed"])
                                 if r["method"] == m and r["target"] == t])
    sp = lambda m, t: np.mean([r["sparsity"] for r in rows if r["method"] == m and r["target"] == t])
    n = len(seeds); tc = T95.get(n - 1, 1.96)
    dense = acc("dense", 0.0)
    summ = dict(n_seeds=n, dense=dict(mean=dense.mean(), std=dense.std(ddof=1)), targets={})
    for t in TARGETS:
        mg, ty = acc("magnitude", t), acc("taylor", t)
        d = ty - mg  # paired by seed (same init and batch order)
        se = d.std(ddof=1) / np.sqrt(n)
        summ["targets"][str(t)] = dict(
            achieved_sparsity=dict(magnitude=sp("magnitude", t), taylor=sp("taylor", t)),
            magnitude=dict(mean=mg.mean(), std=mg.std(ddof=1)),
            taylor=dict(mean=ty.mean(), std=ty.std(ddof=1)),
            paired_diff_taylor_minus_magnitude=dict(mean=d.mean(), ci95=[d.mean() - tc * se, d.mean() + tc * se],
                                                    t_stat=float(d.mean() / se) if se > 0 else None,
                                                    taylor_wins=int((d > 0).sum()), ties=int((d == 0).sum())))
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summ, f, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axhline(dense.mean() * 100, color="#555", ls="--", lw=1, label=f"dense ({dense.mean()*100:.2f}%)")
    for m, c in (("magnitude", "#1f77b4"), ("taylor", "#d62728")):
        mu = [acc(m, t).mean() * 100 for t in TARGETS]; sd = [acc(m, t).std(ddof=1) * 100 for t in TARGETS]
        ax.errorbar([sp(m, t) * 100 for t in TARGETS], mu, yerr=sd, marker="o", capsize=3, color=c,
                    label=f"{m} (mean ± sd, {n} seeds)")
    ax.set_xlabel("achieved weight sparsity (%)"); ax.set_ylabel("test accuracy (%)")
    ax.set_title("Digits MLP 64-256-128-10: accuracy vs sparsity"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/accuracy_vs_sparsity.png", dpi=150)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for r in rows:
        if r["seed"] == 0 and (r["method"] == "dense" or r["target"] == 0.95):
            c = np.array(r["curve"]); lab = r["method"] if r["method"] == "dense" else f"{r['method']} 95%"
            ax[0].plot(c[:, 0], c[:, 1], label=lab); ax[1].plot(c[:, 0], c[:, 2] * 100, label=lab)
    ax[0].set_yscale("log"); ax[0].set_title("train loss (seed 0)"); ax[1].set_title("test accuracy % (seed 0)")
    for x in ax: x.set_xlabel("epoch"); x.grid(alpha=.3); x.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/training_curves.png", dpi=150)
    print(json.dumps(summ, indent=2, default=float))


if __name__ == "__main__":
    main()
