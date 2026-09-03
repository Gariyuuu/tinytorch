"""Multi-class classification on an MNIST-like 8x8 digit set.

The data is *synthetic*: procedurally drawn seven-segment glyphs with noise and
a one-pixel jitter (see ``datasets.make_digits``). That keeps the repository
download-free while exercising exactly the machinery a real MNIST run would --
mini-batching, softmax cross-entropy over 10 classes, a held-out split, and
train/eval mode switching around dropout.

Accuracy here says something about TinyTorch, not about real MNIST.

Run:  python examples/03_digits_classification.py
"""

from __future__ import annotations

import time

import numpy as np
from datasets import make_digits, train_test_split

import tinytorch as tt
from tinytorch import nn, optim

EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 5e-3


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return ``(loss, accuracy)`` in eval mode, without building a graph."""
    model.eval()
    with tt.no_grad():
        logits = model(tt.Tensor(x.astype(np.float64)))
        loss = nn.CrossEntropyLoss()(logits, tt.Tensor(y)).item()
        accuracy = float((logits.data.argmax(axis=1) == y).mean())
    model.train()
    return loss, accuracy


def render(image: np.ndarray) -> str:
    """Draw one 8x8 sample as ASCII art."""
    ramp = " .:-=+*#%@"
    pixels = image.reshape(8, 8)
    return "\n".join(
        "    " + "".join(ramp[min(int(v * len(ramp)), len(ramp) - 1)] for v in row)
        for row in pixels
    )


def main() -> None:
    x, y = make_digits(n_per_class=100, noise=0.15, seed=3)
    x_train, y_train, x_test, y_test = train_test_split(x, y, test_fraction=0.25, seed=4)
    print(
        f"train: {x_train.shape[0]} samples   test: {x_test.shape[0]} samples   "
        f"features: {x_train.shape[1]}   classes: 10"
    )
    print(f"\na sample (label = {y_train[0]}):")
    print(render(x_train[0]))

    tt.manual_seed(6)
    model = nn.Sequential(
        nn.Linear(64, 48),
        nn.ReLU(),
        nn.Linear(48, 10),
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    print(f"\n{model}\nparameters: {model.num_parameters()}\n")

    rng = np.random.default_rng(7)
    print(
        f"{'epoch':>6}  {'train loss':>11}  {'train acc':>10}  "
        f"{'test loss':>10}  {'test acc':>9}"
    )
    print("-" * 56)

    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(x_train))
        for begin in range(0, len(order), BATCH_SIZE):
            index = order[begin : begin + BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(tt.Tensor(x_train[index].astype(np.float64)))
            loss_fn(logits, tt.Tensor(y_train[index])).backward()
            optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            train_loss, train_accuracy = evaluate(model, x_train, y_train)
            test_loss, test_accuracy = evaluate(model, x_test, y_test)
            print(
                f"{epoch:>6}  {train_loss:>11.5f}  {train_accuracy:>10.4f}  "
                f"{test_loss:>10.5f}  {test_accuracy:>9.4f}"
            )
    elapsed = time.perf_counter() - start

    _, final_accuracy = evaluate(model, x_test, y_test)
    print(f"\ntrained {EPOCHS} epochs in {elapsed:.2f}s")
    print(f"final held-out accuracy: {final_accuracy:.4f}   (chance is 0.1000)")

    # Per-class breakdown makes it obvious if the model is just predicting one class.
    model.eval()
    with tt.no_grad():
        predicted = model(tt.Tensor(x_test.astype(np.float64))).data.argmax(axis=1)
    print("\nper-class accuracy on the held-out split:")
    for digit in range(10):
        mask = y_test == digit
        if mask.any():
            hit = float((predicted[mask] == digit).mean())
            print(f"  {digit}: {hit:.3f}  ({int(mask.sum())} samples)")


if __name__ == "__main__":
    main()
