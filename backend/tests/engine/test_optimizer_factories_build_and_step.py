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


def test_sophiah_builds_and_steps_with_a_plain_backward():
    """The trainer's path; test_sophiah_trainer_estimate.py covers its estimate."""
    model, loss = _model_and_loss()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = OptimizerFactory.create(
        "SophiaH", model.parameters(), lr=0.001, config={"sophia_update_period": 2}
    )
    assert type(optimizer).__name__ == "SophiaH"

    for _ in range(4):
        optimizer.zero_grad()
        loss().backward()
        optimizer.step()

    _assert_moved_and_finite(model, before)


def _sophiag(model):
    optimizer = OptimizerFactory.create("SophiaG", model.parameters(), lr=0.001)
    assert type(optimizer).__name__ == "SophiaG"
    return optimizer


def test_sophiag_is_the_real_class_and_steps_with_a_plain_backward():
    model, loss = _model_and_loss()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = _sophiag(model)

    for _ in range(4):
        optimizer.zero_grad()
        loss().backward()
        optimizer.step()

    _assert_moved_and_finite(model, before)


def test_sophiag_state_survives_a_checkpoint_round_trip():
    model, loss = _model_and_loss()
    optimizer = _sophiag(model)
    for _ in range(2):
        optimizer.zero_grad()
        loss().backward()
        optimizer.step()

    resumed = _sophiag(model)
    resumed.load_state_dict(optimizer.state_dict())
    for param in model.parameters():
        assert torch.equal(
            resumed.state[param]["hessian"], optimizer.state[param]["hessian"]
        )

    before = [p.detach().clone() for p in model.parameters()]
    resumed.zero_grad()
    loss().backward()
    resumed.step()
    _assert_moved_and_finite(model, before)


def test_sophiag_fails_by_name_when_its_package_cannot_load(monkeypatch):
    """Never another optimizer in its place: the user selected SophiaG."""
    from app.engine.factories.optimizers import sophiag

    monkeypatch.setattr(sophiag.importlib.util, "find_spec", lambda _name: None)
    model, _ = _model_and_loss()
    with pytest.raises(RuntimeError, match="SophiaG"):
        OptimizerFactory.create("SophiaG", model.parameters(), lr=0.001)


def test_sophiag_resumes_a_checkpoint_of_the_old_silent_adamw_run(monkeypatch):
    """SophiaG jobs used to train with AdamW, so their checkpoints hold AdamW's state.
    The run keeps its weights and starts SophiaG's state fresh, and says so."""
    from app.engine.factories.optimizers import sophiag

    warnings = []
    monkeypatch.setattr(
        sophiag.logger, "warning", lambda event, **_: warnings.append(event)
    )
    model, loss = _model_and_loss()
    old = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss().backward()
    old.step()
    checkpoint = old.state_dict()

    resumed = _sophiag(model)
    resumed.load_state_dict(checkpoint)

    assert warnings == ["sophiag_ignored_other_optimizers_state"]
    before = [p.detach().clone() for p in model.parameters()]
    resumed.zero_grad()
    loss().backward()
    resumed.step()
    _assert_moved_and_finite(model, before)


@pytest.mark.xdist_group("gpu")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="capturable needs a CUDA device")
def test_sophiag_capturable_steps_with_the_trainer_s_float_learning_rate():
    torch.manual_seed(0)
    x = torch.randn(16, 4, device="cuda")
    model = torch.nn.Linear(4, 3).to("cuda")
    optimizer = OptimizerFactory.create(
        "SophiaG", model.parameters(), lr=0.001, config={"sophia_capturable": True}
    )
    before = [p.detach().clone() for p in model.parameters()]

    optimizer.zero_grad()
    model(x).pow(2).mean().backward()
    optimizer.step()

    assert any(not torch.equal(p, b) for p, b in zip(model.parameters(), before))
    assert isinstance(optimizer.param_groups[0]["lr"], float), "the scheduler owns a float"
