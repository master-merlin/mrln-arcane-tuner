"""The Hessian diagonal estimate SophiaH trains on, from ordinary gradient passes.

No test here builds a graph for the estimator: PyTorch has no second derivative
for the attention the trainer's models use, so the estimator must never need one.
"""
import random

import numpy as np
import pytest
import torch

from app.engine.core.pipeline.hessian_estimate import estimate_hessian_diagonal


def _quadratic():
    """f(x) = 0.5 x'Ax with a known Hessian A, off-diagonal terms included."""
    torch.manual_seed(0)
    root = torch.randn(6, 6)
    a = root @ root.T + torch.eye(6)
    x = torch.nn.Parameter(torch.randn(6))
    return a, x, lambda: [a @ x]


def _network():
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3)
    )
    data = torch.randn(32, 4)
    target = torch.randn(32, 3)
    return (
        list(model.parameters()),
        lambda: torch.nn.functional.mse_loss(model(data), target),
    )


@pytest.mark.parametrize("distribution", ["gaussian", "rademacher"])
def test_the_estimate_averages_to_the_true_diagonal(distribution):
    a, x, grads = _quadratic()
    estimate = estimate_hessian_diagonal(
        [x], grads, num_samples=4000, distribution=distribution
    )
    relative = (estimate[0] - a.diagonal()).abs() / a.diagonal()
    assert relative.max() < 0.25


def test_one_sample_matches_the_exact_product_on_a_real_network():
    params, loss = _network()

    torch.manual_seed(7)
    zs = [torch.randn_like(p) for p in params]
    grads = torch.autograd.grad(loss(), params, create_graph=True)
    products = torch.autograd.grad(grads, params, grad_outputs=zs)
    exact = torch.cat([(z * h).reshape(-1) for z, h in zip(zs, products)])

    torch.manual_seed(7)
    estimate = estimate_hessian_diagonal(
        params, lambda: torch.autograd.grad(loss(), params)
    )
    found = torch.cat([e.reshape(-1) for e in estimate])

    assert torch.nn.functional.cosine_similarity(found, exact, dim=0) > 0.99


def test_parameters_are_bitwise_unchanged_and_grads_untouched():
    params, loss = _network()
    loss().backward()
    before = [p.detach().clone() for p in params]
    grads_before = [p.grad.clone() for p in params]

    estimate_hessian_diagonal(params, lambda: torch.autograd.grad(loss(), params))

    assert all(torch.equal(p, b) for p, b in zip(params, before))
    assert all(torch.equal(p.grad, g) for p, g in zip(params, grads_before))


@pytest.mark.parametrize("fail_on_pass", [1, 2])
def test_parameters_are_restored_when_a_pass_raises(fail_on_pass):
    params, loss = _network()
    before = [p.detach().clone() for p in params]
    passes = []

    def grads():
        passes.append(1)
        if len(passes) == fail_on_pass:
            raise RuntimeError("out of memory")
        return torch.autograd.grad(loss(), params)

    with pytest.raises(RuntimeError, match="out of memory"):
        estimate_hessian_diagonal(params, grads)

    assert all(torch.equal(p, b) for p, b in zip(params, before))


def test_rademacher_samples_both_signs():
    """The library's own sampler yields only -1, so its estimate has one sign of z."""
    x = torch.nn.Parameter(torch.zeros(64))
    seen = []

    def grads():
        seen.append(x.detach().clone())
        return [torch.zeros(64)]

    estimate_hessian_diagonal([x], grads, distribution="rademacher")
    assert (seen[0] > 0).any() and (seen[0] < 0).any()


def test_an_unknown_distribution_is_refused():
    _, x, grads = _quadratic()
    with pytest.raises(ValueError, match="uniform"):
        estimate_hessian_diagonal([x], grads, distribution="uniform")


def test_a_parameter_the_pass_did_not_reach_gets_a_zero_estimate():
    a, x, _ = _quadratic()
    unused = torch.nn.Parameter(torch.ones(3))

    estimate = estimate_hessian_diagonal([x, unused], lambda: [a @ x, None])

    assert torch.equal(estimate[1], torch.zeros(3))
    assert estimate[0].abs().sum() > 0


def test_both_passes_of_a_pair_see_the_same_noise():
    """Different noise in the two passes would swamp the small difference."""
    a, x, _ = _quadratic()

    def noisy_grads():
        return [a @ x + 100.0 * torch.randn(6)]

    torch.manual_seed(3)
    z = torch.randn(6)
    torch.manual_seed(3)
    estimate = estimate_hessian_diagonal([x], noisy_grads)

    assert torch.allclose(estimate[0], z * (a @ z), rtol=0.01, atol=0.05)


def test_the_training_random_streams_continue_as_if_it_never_ran():
    _, x, grads = _quadratic()

    def draw_all():
        torch.randn(4)
        random.random()
        np.random.random()
        return grads()

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    expected = (torch.rand(1).item(), random.random(), np.random.random())

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    estimate_hessian_diagonal([x], draw_all)

    assert (torch.rand(1).item(), random.random(), np.random.random()) == expected


def test_a_non_finite_pass_gives_no_estimate():
    x = torch.nn.Parameter(torch.ones(3))
    assert (
        estimate_hessian_diagonal([x], lambda: [torch.full((3,), float("inf"))]) is None
    )
