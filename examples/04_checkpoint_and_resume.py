"""Save a model and an optimizer mid-training, then resume exactly.

This demonstrates the thing that makes optimizer state real state: Adam's
moment estimates and step counter must survive the round trip, or the resumed
run diverges from the uninterrupted one. The example proves it by running both
and comparing parameters bit for bit.

Run:  python examples/04_checkpoint_and_resume.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from datasets import xor_data

import tinytorch as tt
from tinytorch import nn, optim


def build():
    tt.manual_seed(21)
    model = nn.Sequential(nn.Linear(2, 12), nn.Tanh(), nn.Linear(12, 2))
    return model, optim.Adam(model.parameters(), lr=0.02)


def train(model, optimizer, x, y, steps):
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    return loss.item()


def main() -> None:
    features, labels = xor_data(n_per_quadrant=60, seed=1)
    x = tt.Tensor(features.astype(np.float64))
    y = tt.Tensor(labels)

    # Reference: 200 uninterrupted steps.
    reference, reference_opt = build()
    reference_loss = train(reference, reference_opt, x, y, 200)
    print(f"uninterrupted run  : 200 steps, final loss {reference_loss:.8f}")

    # Interrupted: 100 steps, checkpoint, rebuild from scratch, 100 more.
    partial, partial_opt = build()
    train(partial, partial_opt, x, y, 100)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.ttz"
        tt.save(
            {
                "model": partial.state_dict(),
                "optimizer": partial_opt.state_dict(),
                "step": 100,
                "note": "halfway through XOR training",
            },
            path,
        )
        print(f"\nwrote {path.name}: {path.stat().st_size} bytes")

        checkpoint = tt.load(path)
        print(f"  keys        : {list(checkpoint)}")
        print(f"  step        : {checkpoint['step']}")
        print(f"  note        : {checkpoint['note']}")
        print(f"  model keys  : {list(checkpoint['model'])}")
        print(f"  adam state  : {sorted(checkpoint['optimizer']['state']['0'])}")

        resumed, resumed_opt = build()
        resumed.load_state_dict(checkpoint["model"])
        resumed_opt.load_state_dict(checkpoint["optimizer"])
        resumed_loss = train(resumed, resumed_opt, x, y, 100)

    print(f"\nresumed run        : 100 + 100 steps, final loss {resumed_loss:.8f}")
    worst = max(
        float(np.abs(a.data - b.data).max())
        for (_, a), (_, b) in zip(
            reference.named_parameters(), resumed.named_parameters(), strict=True
        )
    )
    print(f"largest parameter difference: {worst:.3e}")
    if worst == 0.0:
        print("MATCH")
    else:
        print(f"MISMATCH (loss delta {abs(reference_loss - resumed_loss):.3e})")

    # Determinism: saving the same state twice yields identical bytes.
    with tempfile.TemporaryDirectory() as directory:
        a, b = Path(directory) / "a.ttz", Path(directory) / "b.ttz"
        tt.save(reference.state_dict(), a)
        tt.save(reference.state_dict(), b)
        print(f"byte-identical re-save: {a.read_bytes() == b.read_bytes()}")


if __name__ == "__main__":
    main()
