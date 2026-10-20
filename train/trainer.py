"""Mini-batch training with optional progressive pruning."""
import numpy as np
from engine.tensor import Tensor, softmax_cross_entropy
from nn.layers import mlp
from optim.adam import Adam
from prune.pruner import Pruner, sparsity
from train.data import load

DEFAULTS = dict(sizes=(64, 256, 128, 10), epochs=40, batch=64, lr=2e-3,
                method=None, target=0.0, prune_start=5, prune_end=30, prune_every=1)  # epochs


def evaluate(model, X, y):
    logits = model(Tensor(X))
    return float(softmax_cross_entropy(logits, y).data), float((logits.data.argmax(1) == y).mean())


def run(seed=0, verbose=False, **kw):
    cfg = {**DEFAULTS, **kw}
    Xtr, ytr, Xte, yte = load(split_seed=0)  # fixed split; `seed` controls init + batch order
    rng = np.random.default_rng(seed)
    model = mlp(list(cfg["sizes"]), rng)
    opt = Adam(model.parameters(), lr=cfg["lr"])
    spe = int(np.ceil(len(ytr) / cfg["batch"]))  # steps per epoch
    pruner = None
    if cfg["method"]:
        pruner = Pruner(model.linears(), opt, cfg["method"], cfg["target"],
                        cfg["prune_start"] * spe, cfg["prune_end"] * spe, cfg["prune_every"] * spe)
    hist, step = [], 0
    for ep in range(cfg["epochs"]):
        perm, tl = rng.permutation(len(ytr)), []
        for i in range(0, len(ytr), cfg["batch"]):
            b = perm[i:i + cfg["batch"]]
            if pruner:
                pruner.maybe_prune(step)
            opt.zero_grad()
            loss = softmax_cross_entropy(model(Tensor(Xtr[b])), ytr[b])
            loss.backward()
            if not np.isfinite(loss.data):
                raise FloatingPointError(f"non-finite loss at step {step}")
            if pruner:
                pruner.observe()
            opt.step()
            tl.append(float(loss.data)); step += 1
        tr_loss, tr_acc = evaluate(model, Xtr, ytr)
        te_loss, te_acc = evaluate(model, Xte, yte)
        h = dict(epoch=ep + 1, train_loss=tr_loss, train_acc=tr_acc, test_loss=te_loss,
                 test_acc=te_acc, sparsity=float(sparsity(model.linears())))
        hist.append(h)
        if verbose:
            print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in h.items()))
    return model, hist
