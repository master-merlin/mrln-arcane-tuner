"""Every pytorch_optimizer the app loads builds through the factory and takes real steps.

An unknown key silently falls back to AdamW, so each test names the class it got.
"""
import pytest
import torch

from app.engine.factories.optimizer import OptimizerFactory


def _model_and_loss():
    torch.manual_seed(0)
    x = torch.randn(16, 4)
    y = x @ torch.randn(4, 3)
    model = torch.nn.Linear(4, 3)
    return model, lambda: torch.nn.functional.mse_loss(model(x), y)


def _assert_moved_and_finite(model, before):
    after = list(model.parameters())
    assert all(torch.isfinite(p).all() for p in after)
    assert any(not torch.equal(p, b) for p, b in zip(after, before))


@pytest.mark.parametrize(
    "key",
    ["Adafactor", "AdEMAMix", "Lion", "RAdam", "Shampoo", "StableAdamW"],
)
def test_optimizer_builds_and_steps(key):
    model, loss = _model_and_loss()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = OptimizerFactory.create(key, model.parameters(), lr=0.001)
    assert type(optimizer).__name__.lower() == key.lower()

    # RAdam takes no step until its warm-up has passed.
    for _ in range(8):
        optimizer.zero_grad()
        loss().backward()
        optimizer.step()

    _assert_moved_and_finite(model, before)


def test_sophiah_steps_and_fills_its_hessian():
    """Only with the graph this test builds: the trainer's backward() builds none."""
    model, loss = _model_and_loss()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = OptimizerFactory.create(
        "SophiaH", model.parameters(), lr=0.001, config={"sophia_update_period": 2}
    )
    assert type(optimizer).__name__ == "SophiaH"

    for _ in range(4):
        optimizer.zero_grad()
        loss().backward(create_graph=True)
        optimizer.step()

    for param in model.parameters():
        assert torch.isfinite(optimizer.state[param]["hessian"]).all()
    _assert_moved_and_finite(model, before)
