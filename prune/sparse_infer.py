"""Sparse-aware inference that never touches inactive connections.

Two exact transformations of a pruned Sequential MLP (Linear / elementwise activation):
 1. Unit compaction: a hidden unit with no incoming active weights outputs the constant act(b_j);
    its contribution act(b_j) * W_next[j, :] is folded into the next bias and the unit is removed.
    A unit with no outgoing active weights is unused and removed too. Iterated to a fixed point.
 2. Each remaining weight matrix is stored in a column-grouped COO/CSC form and applied with a
    gather kernel whose work is O(batch * nnz), not O(batch * fan_in * fan_out).
"""
import numpy as np
from nn.layers import Linear


class SparseLinear:
    def __init__(self, W, b):
        rows, cols = np.nonzero(W)
        o = np.lexsort((rows, cols))  # group by output column
        self.rows, self.cols, self.vals = rows[o], cols[o], W[rows[o], cols[o]]
        self.ucols, self.starts = np.unique(self.cols, return_index=True)
        self.b, self.fan_out, self.nnz = b.copy(), W.shape[1], len(self.vals)

    def __call__(self, x):
        out = np.broadcast_to(self.b, (x.shape[0], self.fan_out)).copy()
        if self.nnz:
            prod = x[:, self.rows] * self.vals                   # B x nnz multiplies
            out[:, self.ucols] += np.add.reduceat(prod, self.starts, axis=1)  # B x nnz adds
        return out


def compact(model):
    """Return [(W, b, act_fn_or_None), ...] with dead units removed (exact)."""
    blocks, layers = [], model.layers
    for i, l in enumerate(layers):
        if isinstance(l, Linear):
            act = layers[i + 1].np_fn if i + 1 < len(layers) and not isinstance(layers[i + 1], Linear) else None
            blocks.append([l.W.data * l.W.mask, l.b.data.copy(), act])
    changed = True
    while changed:
        changed = False
        for k in range(len(blocks) - 1):
            W, b, act = blocks[k]
            Wn, bn, _ = blocks[k + 1]
            no_in = ~W.any(axis=0)
            no_out = ~Wn.any(axis=1)
            dead = no_in | no_out
            if not dead.any():
                continue
            fold = no_in & ~no_out  # constant outputs that still feed forward
            blocks[k + 1][1] = bn + act(b[fold]) @ Wn[fold]
            keep = ~dead
            blocks[k][0], blocks[k][1] = W[:, keep], b[keep]
            blocks[k + 1][0] = Wn[keep]
            changed = True
    return blocks


class SparseMLP:
    def __init__(self, model):
        self.blocks = [(SparseLinear(W, b), act) for W, b, act in compact(model)]

    def __call__(self, x):
        for lin, act in self.blocks:
            x = lin(x)
            if act is not None:
                x = act(x)
        return x

    def macs(self):
        return sum(lin.nnz for lin, _ in self.blocks)

    def nbytes(self):  # float64 values + int64 row index + per-column offsets + bias
        return sum(l.vals.nbytes + l.rows.nbytes + l.ucols.nbytes + l.starts.nbytes + l.b.nbytes
                   for l, _ in self.blocks)


class CompactDenseMLP:
    """Dense BLAS matmuls on the compacted (smaller) matrices - a second honest sparse path."""

    def __init__(self, model):
        self.blocks = compact(model)

    def __call__(self, x):
        for W, b, act in self.blocks:
            x = x @ W + b
            if act is not None:
                x = act(x)
        return x

    def macs(self):
        return sum(W.size for W, _, _ in self.blocks)
