# Design

Notation: loss L(w); a weight w_i; minibatch gradient g_b = ∂L_b/∂w; mask a_i ∈ {0,1};
effective weight w_eff = a ⊙ w.

## 1. Importance criterion

**Goal.** Rank connections by ΔL_i, the change in loss when w_i is set to 0
(perturbation δ = −w_i e_i).

**Taylor expansion.**
ΔL_i = L(w − w_i e_i) − L(w) ≈ −g_i w_i + ½ H_ii w_i² (first + diagonal second order).

* Magnitude pruning (baseline) assumes ΔL_i is monotone in |w_i|, i.e. that all connections have
  the same curvature and gradient — ignoring how the loss actually depends on each weight.
* A pure first-order score |g_i w_i| fails later in training: near a minimum the full-batch gradient
  g → 0, so the first-order term vanishes while removing weights still hurts.

**Score used.** s_i = E_b[(g_{b,i} w_i)²], averaged over all minibatches since the previous
pruning event (`Pruner.observe`). Writing g_b = g + ε_b with E[ε_b] = 0:

  s_i = w_i² (g_i² + Var(g_{b,i})).

* The g_i² w_i² part is the square of the first-order term.
* Var(g_{b,i}) ≈ F_ii / B, where F is the (empirical) Fisher. For a cross-entropy loss near a
  minimum, F ≈ H (Gauss–Newton), so w_i² Var(g_{b,i}) ∝ ½ H_ii w_i², the Optimal-Brain-Damage
  second-order term.

So the score tracks both the first- and second-order contributions to ΔL_i, and needs only the
gradients already computed during training (one extra elementwise multiply-add per step).
This is the criterion of Molchanov et al. (2019), applied to individual weights.

**Assumptions.** (i) Removing a weight is a small perturbation (quadratic model valid);
(ii) off-diagonal Hessian terms are ignored — weights are scored independently, so jointly pruning
correlated weights can cost more than the sum of scores; (iii) empirical Fisher ≈ Hessian, which
holds best near a minimum with a well-fit model; (iv) scores from different layers have different
scales, so ranking is **per layer** (uniform layer-wise sparsity) rather than global.

## 2. Gradient semantics for masked parameters

Forward: `W_eff = W * mask; y = x @ W_eff + b` (the mask is a constant, non-differentiable tensor).

* A masked connection contributes exactly 0 to the forward pass, whatever value is stored in
  `W.data` (tested with a stale value of 1e6).
* **Parameter gradient is zero:** `W.grad = mask ⊙ G` with G = ∂L/∂W_eff. This falls out of the
  chain rule through the `mul` node, so no special-casing is needed in the engine.
* **The dense signal is kept separately:** G is available on `layer.W_eff.grad` after `backward()`.
  This is what `regrow()` uses to rank inactive connections (RigL: large |G| means the loss would
  drop quickly if the connection were reactivated).

## 3. Optimizer state during pruning and reactivation

Adam stores m, v and a step counter **t per element** (not one global t).
For an element with mask a:

```
t ← t + a
m ← a·(β1 m + (1−β1) g)          v ← a·(β2 v + (1−β2) g²)
w ← a·(w − lr · m̂ / (√v̂ + ε)),   m̂ = m/(1−β1^t), v̂ = v/(1−β2^t)
```

* **On prune** (`Pruner.prune_to`): `w ← 0`, and `m, v, t ← 0` for newly pruned entries
  (`Adam.reset_state`). While masked, the update rule keeps all four at exactly 0, so no momentum
  carries over and nothing moves a masked weight.
* **On reactivation** (`regrow`): `w ← 0` and the state is reset to `m = v = t = 0`. The first update
  is then a correctly bias-corrected fresh Adam step, −lr·g/(|g|+ε). With a global step counter
  the bias correction would instead assume a long history that this weight doesn't have, shrinking
  m̂ toward 0 and giving a wrong first step. A per-element t avoids this.
* `tests/test_masking.py::test_prune_update_regrow_regression` runs train → prune → train →
  regrow → step and checks the exact first update of the regrown weights, that t = 1, and that all
  still-masked weights are exactly 0.

**Schedule.** s_t = s_f + (s_0 − s_f)(1 − (t − t_0)/(t_1 − t_0))³ (Zhu & Gupta 2017), applied once
per epoch from epoch 5 to 30, followed by 10 epochs of fine-tuning. The cubic schedule prunes
quickly while redundancy is high and slowly near the target, so the network can recover between
steps. One-shot pruning to 95–99% removes many weights whose scores were measured in a very
different network; the gradual schedule re-estimates scores on the network that is left.
The trade-off is a longer training run and three extra hyperparameters (t_0, t_1, Δt).
Pruning is monotone: once removed, a weight stays removed unless `regrow` is called.

## 4. Engine bottlenecks and possible optimizations

Measured with `cProfile` (5 epochs, batch 64, 1 thread): **`Adam.step` takes ~50% of training
time**, matmul forward + backward ~20%, and graph bookkeeping (`backward`, closures, ReLU, add) the rest.

1. **Mask-aware Adam.** The per-element bias correction calls `β**t` elementwise on every
   parameter every step, plus several `np.where` passes. *Fix:* when all active entries share
   the same t (always true without regrowth), use the scalar correction. Otherwise look up
   `1−β^t` from a precomputed table indexed by t. Also fuse the updates in place.
2. **Python graph overhead.** Each op allocates a `Tensor` and a closure; the topological sort runs
   on every `backward()`. That is fine at ~15 nodes, but it dominates for small tensors or deep graphs.
   *Fix:* trace once and replay a static tape, or fuse common patterns (Linear+bias+ReLU) into one node.
3. **Memory traffic.** Every op allocates a new array; `_unbroadcast` and `sum` allocate temporaries;
   intermediate `.grad`s are kept for every node. *Fix:* release intermediate grads after use, use
   in-place accumulation into preallocated buffers, and use float32 (about 2× less bandwidth).
4. **Masking is dense.** `W * mask` and the dense matmul cost the same at 98% sparsity.
   *Fix:* once sparsity is high, switch training to compacted matrices (as in inference) or a
   compiled sparse kernel (CSR SpMM, block sparsity for SIMD).
5. **Single thread / no batching across steps.** *Fix:* multithreaded BLAS (disabled in the
   benchmark for fair timing), and data loading that overlaps with compute.
6. **Unstructured sparse kernel in NumPy.** Gather + `reduceat` allocate an nnz×B intermediate and
   run at a small fraction of BLAS throughput (see README). *Fix:* a compiled CSR/ELL kernel
   (C/Numba/Cython), 2:4 or block sparsity that maps onto hardware, or more structured pruning so
   compaction removes more.

## 5. Production inference service for sparse models

**Model artifact.** Serving doesn't run the autodiff engine. Export compacted, immutable artifacts:
per-layer format chosen offline by benchmarking (compact-dense vs. CSR vs. block-sparse), float32
or int8 values, plus a manifest (version, input schema, sparsity, checksum, reference outputs for
a golden batch). On load, run the golden batch and refuse to serve if the outputs don't match.

**Isolation.** Each model version runs in its own worker process pool (or container), with
read-only weights mmapped from the artifact, so pools of the same version share pages.
One model's crash, OOM or slow kernel can't affect others. Versions are side-by-side for
canary/shadow traffic. Rollback is a routing change.

**Resource management.** Per-model CPU quotas (cgroups) and memory limits sized from the artifact's
measured peak. Pin BLAS threads per worker (threads × workers ≤ cores) to avoid oversubscription.
Admission control uses a bounded queue per model; when it is full, reject fast (429) rather than
let latency grow without bound. Autoscale on queue depth and p95 latency, not CPU alone.

**Latency vs. throughput.** A dynamic micro-batcher per model collects requests up to `max_batch`
or `max_wait` (e.g. 2 ms). The benchmark shows why this matters: at batch 1, per-call overhead
dominates (~10 µs regardless of sparsity), while at batch 1024 compaction gives 2.5–4.5×.
Separate latency-sensitive traffic (small `max_wait`) from bulk traffic (large batches) with
priority queues. Choose the kernel per batch-size bucket from offline benchmarks, since the
fastest format depends on batch size and sparsity.

**Concurrency model.** An async front end (HTTP/gRPC) handles request parsing and validation
and enqueues to the model's batcher. Compute workers are separate processes to avoid the GIL.
Requests carry deadlines, and expired requests are dropped before compute.

**Monitoring.** Per model/version: request rate, error rate, p50/p95/p99 latency split into queue
time and compute time, batch-size histogram, queue depth, rejections, CPU/memory per worker.
Model health: prediction-class distribution and confidence drift vs. the training distribution,
input-feature range checks, and periodic golden-batch replays to detect numerical regressions.
Export as Prometheus metrics with alerts on SLO burn rate, plus traces sampled per request.
Log artifact version on every response for auditability.
