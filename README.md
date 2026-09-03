# TinyTorch

A small, readable deep-learning framework built from scratch on NumPy.

TinyTorch implements **reverse-mode automatic differentiation**, a dynamic
computation graph, a `Module`/`Parameter` neural-network API, optimizers with
real state, and a safe deterministic checkpoint format — all of it its own.

> **This is not a PyTorch wrapper.** NumPy supplies storage and array
> arithmetic; it supplies no derivative information at all. Every derivative
> rule, the graph, the topological backward traversal, gradient accumulation and
> broadcast gradient reduction are implemented here. PyTorch appears only as a
> *test* dependency, used to verify the results.

**Verified:** 391 tests. Every operator checked against finite differences, and
against PyTorch at `rtol=1e-10` in float64 — forward outputs, losses, gradients
and optimizer trajectories over 25 steps. On the completion test, agreement is
at machine epsilon (~1e-16).

---

## Install

```bash
git clone https://github.com/Gariyuuu/tinytorch && cd tinytorch
python -m venv .venv && source .venv/bin/activate

pip install -e .              # runtime: numpy only
pip install -e ".[dev]"       # + pytest, torch (tests only), ruff, mypy
```

Requires Python 3.10+.

## Quick start

```python
import numpy as np
import tinytorch as tt
from tinytorch import nn, optim

tt.manual_seed(0)

model = nn.Sequential(
    nn.Linear(4, 32), nn.ReLU(),
    nn.Linear(32, 3),
)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

x = tt.randn(128, 4)
y = tt.Tensor(np.random.default_rng(0).integers(0, 3, 128))

for epoch in range(100):
    optimizer.zero_grad()          # backward accumulates — this is required
    loss = loss_fn(model(x), y)
    loss.backward()                # one pass fills every parameter's .grad
    optimizer.step()

tt.save(model.state_dict(), "model.ttz")
```

Autograd on its own:

```python
x = tt.Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2 + 3 * x).sum()
y.backward()
x.grad                             # array([ 5.,  7.,  9.])  ==  2x + 3
```

## What is implemented

**Autograd** — `Function`/`Context`/`Node` abstraction, dynamic define-by-run
graph, iterative topological backward traversal, gradient accumulation,
`requires_grad` propagation, `detach`, `retain_grad`, `zero_grad`,
`no_grad`/`enable_grad` (thread-local), broadcasting-aware gradient reduction,
and user-definable primitives.

**Operators** — `add` `sub` `mul` `div` `pow` `neg` `matmul` `sum` `mean` `max`
`reshape` `transpose` `getitem` `exp` `log` `abs` `clone` `maximum` `minimum`
`relu` `sigmoid` `tanh` `gelu` `softmax` `log_softmax`. All broadcast; all
verified numerically and against PyTorch.

**Neural networks** — `Module`, `Parameter`, `Linear`, `Sequential`, `Dropout`,
`Flatten`, `Identity`; activations `ReLU` `LeakyReLU` `Sigmoid` `Tanh` `GELU`
`Softmax` `LogSoftmax`; losses `MSELoss` `L1Loss` `NLLLoss` `CrossEntropyLoss`
`BCELoss`; `parameters()` `named_parameters()` `buffers()` `train()` `eval()`
`state_dict()` `load_state_dict()` `zero_grad()` `apply()`.

**Initialization** — `zeros_` `ones_` `constant_` `uniform_` `normal_`
`xavier_uniform_` `xavier_normal_` `kaiming_uniform_` `kaiming_normal_`,
`calculate_gain`, fan computation.

**Optimizers** — `SGD` (momentum, Nesterov, dampening, weight decay), `Adam`
(bias correction, weight decay, amsgrad), `AdamW` (decoupled decay), parameter
groups, lazily created per-parameter state, and `state_dict`/`load_state_dict`.

**Serialization** — deterministic, pickle-free `.ttz` format (see below).

## Running things

```bash
pytest                                    # 391 tests, ~25s
pytest -m "not slow"                      # skip the training-loop tests
pytest tests/test_torch_parity.py -v      # the PyTorch comparison

ruff check .                              # lint
mypy                                      # type check

cd examples
python 01_linear_regression.py            # fit y = Xw + b, check against lstsq
python 02_xor_classification.py           # MLP solves XOR; linear model cannot
python 03_digits_classification.py        # 8x8 MNIST-like, 10 classes
python 04_checkpoint_and_resume.py        # save/restore mid-training, exactly
python 05_custom_function.py              # add your own differentiable op
python 06_pytorch_equivalence.py          # the completion test

python benchmarks/bench_overhead.py       # overhead characterisation
```

## How it works

`Function.apply` is where the graph is built. It unwraps tensors to raw arrays,
runs `forward`, wraps the result, and — if any input requires gradient — records
a `Node` on the output:

```python
class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)   # backward needs both operands
        return a * b

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        return g * b, g * a           # ∂(ab)/∂a = b,  ∂(ab)/∂b = a
```

`backward()` then walks the recorded DAG in reverse topological order,
multiplying local vector-Jacobian products together — the chain rule — and
summing contributions wherever a tensor had more than one consumer.

Two properties fall out of doing it in one place:

- **Broadcasting is free for every op.** Broadcasting copies data, a copy is a
  fan-out, and the adjoint of a fan-out is a sum. `unbroadcast` performs that
  reduction in the engine, so no operator implements it.
- **Reuse is free.** A tensor with several consumers is visited once, after all
  of them have contributed, and their gradients are summed — the multivariable
  chain rule falling out of the traversal order.

Full explanation with diagrams: **[docs/autograd.md](docs/autograd.md)** and
**[docs/architecture.md](docs/architecture.md)**.

## Validation

Correctness is established three independent ways.

### 1. Finite differences (framework-independent)

```python
x = tt.randn(4, 5, requires_grad=True, dtype=np.float64)
tt.gradcheck(lambda t: tt.log_softmax(t * 2).sum(), [x])   # True or raises
```

Central differences, `eps=1e-6`, `rtol=1e-5`. `gradcheck` **refuses float32**:
the error of a central difference is `O(h²) + O(ε/h)`, and float32's
`ε ≈ 1.2e-7` leaves no usable window for `h`. A deliberately wrong derivative is
included in the suite to prove the checker actually catches one.

### 2. PyTorch parity

84 tests build the same computation in both frameworks from bit-identical
inputs and compare in float64 at `rtol=1e-10, atol=1e-12` — tight enough that
only a genuine formula difference could fail it. This catches conventions
finite differences cannot see: `relu'(0) = 0`, `torch.maximum`'s 0.5/0.5 tie
split, SGD seeding its momentum buffer with the first gradient rather than
zeros, and Adam's exact bias-correction ordering.

### 3. Real training

Convergence is asserted against known answers — recovered generating
parameters, the `lstsq` solution, and a held-out accuracy floor. A control test
confirms a linear model *fails* XOR, so the MLP's success is attributable to
the nonlinearity.

### Test counts

| file | tests | covers |
|---|---:|---|
| `test_gradcheck.py` | 101 | every operator vs finite differences |
| `test_torch_parity.py` | 84 | ops, losses, layers, networks, optimizers vs PyTorch |
| `test_nn.py` | 45 | discovery, train/eval, state dict, layers, init |
| `test_tensor.py` | 44 | construction, dtype policy, operators, conversion |
| `test_autograd.py` | 36 | graph shape, accumulation, detach, grad mode |
| `test_optim.py` | 36 | update rules by hand, state lifecycle, groups |
| `test_serialization.py` | 28 | round-trip, determinism, safety, error handling |
| `test_training.py` | 17 | convergence on real problems |
| **total** | **391** | ~25 s |

Without PyTorch installed, 307 pass and the 84 parity tests skip — CI runs that
configuration to prove NumPy really is the only runtime dependency. Those 307
TinyTorch-only tests complete in **~2 s**; nearly all of the full suite's ~25 s
is PyTorch (a ~2.8 s import plus its own per-call cost across 84 parity tests,
several of which train for 25 steps in both frameworks).

### Completion test

`examples/06_pytorch_equivalence.py` builds a 6→16→12→4 tanh MLP in both
frameworks with identical weights. Measured on the reference machine:

| check | max abs difference |
|---|---|
| forward logits | 4.4e-16 |
| cross-entropy loss | 4.4e-16 |
| parameter gradients (all 6) | ≤ 1.1e-16 |
| after one SGD / SGD+momentum / Adam step | ≤ 2.2e-16 |
| after 10 Adam steps | 2.8e-16 |

That is agreement at machine epsilon — five orders of magnitude inside the
1e-10 tolerance the suite enforces.

## Serialization

`torch.save` and `np.save(allow_pickle=True)` deserialise by executing pickle
opcodes that can import arbitrary modules and call arbitrary callables. Since
weights are routinely downloaded from the internet, TinyTorch does not do that.

A `.ttz` file is a zip containing `manifest.json` plus one `.npy` per array,
written with `allow_pickle=False`. Loading parses JSON and reads arrays; **no
code is executed**, and object-dtype arrays are refused at save time.

Saving the same state twice is **byte-identical** — fixed zip timestamps, fixed
key order, fixed compression level — so checkpoints are hashable and diffable.

```python
tt.save({"model": model.state_dict(),
         "optim": optimizer.state_dict(),
         "epoch": 12}, "run.ttz")

ckpt = tt.load("run.ttz")
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optim"])   # momentum and step counter restored
```

Restoring optimizer state reproduces an interrupted run *exactly*; both the test
suite and `examples/04` assert a parameter difference of exactly 0.0.

## Benchmarks

TinyTorch is **not** trying to be fast. The benchmark characterises overhead so
the cost of a Python autodiff engine is a measured number rather than a guess.
Measured on an M-series Mac, Python 3.11, NumPy 2.4.6, torch 2.14.0 (1 thread),
float64, CPU, best of 5 repeats:

| workload | TinyTorch | PyTorch | ratio |
|---|---:|---:|---:|
| `add`, 1×1 | 0.0062 ms | 0.0095 ms | **0.7×** |
| `matmul`, 32×32 | 0.0227 ms | 0.0239 ms | 1.0× |
| `matmul`, 256×256 | 0.355 ms | 0.341 ms | 1.0× |
| `exp`, 256×256 | 0.313 ms | 0.160 ms | 2.0× |
| training step, 698 params | 0.133 ms | 0.112 ms | 1.2× |
| training step, 2.1 M params | 37.6 ms | 24.2 ms | 1.6× |
| 1000-op chain, fwd+bwd | 6.20 ms | 2.62 ms | 2.4× |

Reading these honestly:

- At 1×1 TinyTorch is *faster* — not a win worth claiming, just that PyTorch's
  full C++ dispatcher costs more per call than a Python path supporting exactly
  one backend. Both are in the same microsecond band.
- Where BLAS does the work (`matmul`) the two are identical, as expected.
- Where a NumPy elementwise kernel plus TinyTorch's extra temporaries do the
  work (`exp`, `relu`) it is ~2× slower — a kernel-quality difference, not an
  autograd-design one.
- The training-step ratio **rises** with size (1.2× → 1.6×), the opposite of the
  usual "overhead amortises away" story: on these shapes TinyTorch was never
  paying much dispatch tax, and what it loses at scale is kernel efficiency.
- Graph traversal cost scales **linearly with graph depth** — ~6 µs/op, flat
  across 10, 100 and 1000 ops — confirming there is no accidental quadratic
  behaviour in the topological walk.

**A correction that changed the numbers.** The first version of this benchmark
constructed the leaf tensors *inside* the timed region. That measured two
tensor constructors rather than two dispatchers, and the constructors differ far
more than the dispatchers do. Leaves are now built once, outside the loop, and
the table above is from the corrected run. The earlier numbers are not reported
anywhere.

A prediction that did not survive contact with the data is worth recording too:
the expected result was the standard "fixed Python overhead amortises away as
models grow." The measurements show the opposite — the ratio *rises* with size.
The generic story is not repeated here, because it is not what happened.

Caveats: CPU and float64 only, single machine, `torch.set_num_threads(1)` but
NumPy's BLAS threading uncontrolled. On a GPU or in float32 at scale the
comparison would not be close.

## Scope and known limitations

**TinyTorch's scope is frozen.** It exists to demonstrate how automatic
differentiation, tensors, neural-network abstractions and optimizers work, in
code short enough to read end to end. It is not a general-purpose training
framework and is not on a path to becoming one.

Everything in the list below is **absent by design, not unfinished**. None of it
is a roadmap, a TODO, or a gap awaiting a contribution. Use PyTorch for real
work; TinyTorch is for understanding what PyTorch is doing.

- **No double backward.** Gradients are `ndarray`, not `Tensor`, so the backward
  pass records no graph. Rules out gradient penalties (WGAN-GP) and MAML-style
  meta-learning.
- **CPU only.** No CUDA, no MPS, no ROCm, no device abstraction of any kind.
  There is no `.to(device)` and there is not meant to be one.
- **No convolutions, normalization or recurrence.** No `Conv2d`, `BatchNorm`,
  `LayerNorm`, `Embedding`, `LSTM` or attention. The op set covers dense
  networks; adding a layer means adding a `Function`, which
  `examples/05_custom_function.py` demonstrates end to end.
- **No in-place tensor mutation on graph tensors.** There is no version counter,
  so mutating a tensor a saved `Context` still references would silently corrupt
  its gradient.
- **`unbroadcast` is applied unconditionally** to every gradient, so an op that
  deliberately returns a differently-shaped gradient has no escape hatch.
- **Exact GELU uses `np.frompyfunc(math.erf)`**, which is accurate to ~1 ulp but
  slow, since NumPy has no vectorised `erf` and SciPy is too heavy a dependency
  for one function. Use `approximate="tanh"` to avoid it.
- **No training infrastructure.** No learning-rate schedulers, no `DataLoader`
  or sampler, no distributed/multi-GPU training, no mixed precision or `autocast`,
  no profiler, no graph visualisation export, no serving or web frontend. The
  examples do their own batching in a plain Python loop, which is the point.
- **Memory is not managed.** Every saved tensor stays alive for the whole
  backward pass; there is no checkpointing or rematerialisation.
- **Benchmarks are from one machine**; ratios travel better than absolute times.

## Repository layout

```
src/tinytorch/
  autograd.py       Function, Context, Node, backward, unbroadcast, no_grad
  tensor.py         Tensor: data, requires_grad, grad, grad_fn, operators
  ops.py            every differentiable primitive + functional API
  creation.py       zeros/ones/randn/... and the seeded global RNG
  gradcheck.py      finite-difference verification
  serialization.py  deterministic, pickle-free .ttz
  nn/               module.py layers.py activations.py losses.py init.py functional.py
  optim/            optimizer.py sgd.py adam.py
tests/              391 tests
examples/           6 runnable scripts + synthetic datasets
benchmarks/         overhead characterisation
docs/               autograd.md, architecture.md
.github/workflows/  lint+types, tests on 3.10-3.13 × Linux/macOS/Windows,
                    a no-PyTorch job, examples, and a build job
```

## License

MIT — see [LICENSE](LICENSE).
