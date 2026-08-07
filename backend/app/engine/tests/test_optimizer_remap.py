"""By-name optimizer-state remap across a rebuild restart (spec §5 step 5).

A rebuild restart recreates the optimizer over only the params that are still
trainable, while the checkpoint's ``optimizer.pt`` covers the wider pre-rebuild
set. Every assertion below is on the state a REAL optimizer ends up holding
after ``load_state_dict`` — not on the shape of the returned dict, which would
pass just as happily with the momentum silently zeroed.
"""

import torch

from app.engine.core.optimization.optimizer_remap import remap_optimizer_state


def _adam_state(params):
    """One AdamW step over ``params`` so every one of them carries momentum."""
    opt = torch.optim.AdamW(params, lr=1e-3)
    for p in params:
        p.grad = torch.ones_like(p)
    opt.step()
    return opt


def test_surviving_params_keep_momentum():
    a, b, c = (torch.nn.Parameter(torch.randn(4, 4)) for _ in range(3))
    saved = _adam_state([a, b, c]).state_dict()
    remapped, missing = remap_optimizer_state(
        saved, saved_names=["m.a", "m.b", "m.c"], current_names=["m.a", "m.c"]
    )
    assert missing == []

    new_opt = torch.optim.AdamW([a, c], lr=1e-3)
    new_opt.load_state_dict(remapped)  # must load without shape/index errors
    loaded = new_opt.state_dict()["state"]
    # The ORIGINAL momentum tensors, positionally re-keyed onto the survivors.
    # Non-zero by construction (one step with grad=1), so a zero-filled
    # "remap" that merely produced the right shape would fail here.
    assert torch.equal(loaded[0]["exp_avg"], saved["state"][0]["exp_avg"])
    assert torch.equal(loaded[1]["exp_avg"], saved["state"][2]["exp_avg"])
    assert loaded[0]["exp_avg"].abs().sum() > 0
    assert loaded[0]["step"] == saved["state"][0]["step"]


def test_dropped_param_state_is_not_carried_onto_a_survivor():
    """Off-by-one insurance: the state of the param that went away must not
    slide onto the one that took its index. A shift is invisible in a shape
    check and poisons every later step with another param's moments."""
    a, b = (torch.nn.Parameter(torch.randn(4, 4)) for _ in range(2))
    b.grad = torch.ones_like(b) * 7.0  # b's moments are distinguishable
    opt = torch.optim.AdamW([a, b], lr=1e-3)
    a.grad = torch.ones_like(a)
    opt.step()
    saved = opt.state_dict()

    remapped, missing = remap_optimizer_state(saved, ["m.a", "m.b"], ["m.b"])
    assert missing == []
    new_opt = torch.optim.AdamW([b], lr=1e-3)
    new_opt.load_state_dict(remapped)
    assert torch.equal(
        new_opt.state_dict()["state"][0]["exp_avg"], saved["state"][1]["exp_avg"]
    )


def test_unmappable_current_param_reported_and_skipped():
    a, b = (torch.nn.Parameter(torch.randn(4, 4)) for _ in range(2))
    saved = _adam_state([a]).state_dict()
    remapped, missing = remap_optimizer_state(
        saved, saved_names=["m.a"], current_names=["m.a", "m.new"]
    )
    # Named, not counted: the caller has to be able to say WHICH param lost its
    # momentum (spec §7 — never a silent zero).
    assert missing == ["m.new"]

    new_opt = torch.optim.AdamW([a, b], lr=1e-3)
    new_opt.load_state_dict(remapped)  # fresh state for m.new, no crash
    assert 1 not in new_opt.state_dict()["state"]


def test_group_level_keys_carry_over():
    a = torch.nn.Parameter(torch.randn(4, 4))
    saved = _adam_state([a]).state_dict()
    saved["param_groups"][0]["d"] = 3.14  # Prodigy-style group state
    remapped, _ = remap_optimizer_state(saved, ["m.a"], ["m.a"])
    assert remapped["param_groups"][0]["d"] == 3.14


def test_group_level_keys_survive_a_narrowing():
    """The narrowing case is the one that matters: Prodigy's d-estimate is the
    run's learned step size, and re-deriving it from scratch after a rebuild
    would visibly restart the LR schedule."""
    a, b = (torch.nn.Parameter(torch.randn(4, 4)) for _ in range(2))
    saved = _adam_state([a, b]).state_dict()
    saved["param_groups"][0]["d"] = 2.5
    saved["param_groups"][0]["lr"] = 1e-5
    remapped, missing = remap_optimizer_state(saved, ["m.a", "m.b"], ["m.a"])
    assert missing == []
    assert remapped["param_groups"][0]["d"] == 2.5
    assert remapped["param_groups"][0]["lr"] == 1e-5
    # Re-indexed onto the surviving params, or load_state_dict rejects it.
    assert remapped["param_groups"][0]["params"] == [0]
    torch.optim.AdamW([a], lr=1e-3).load_state_dict(remapped)


def test_saved_state_is_not_mutated():
    """The caller may fall back to the un-remapped state; remapping in place
    would corrupt what it falls back to."""
    a, b = (torch.nn.Parameter(torch.randn(4, 4)) for _ in range(2))
    saved = _adam_state([a, b]).state_dict()
    before = list(saved["param_groups"][0]["params"])
    remap_optimizer_state(saved, ["m.a", "m.b"], ["m.a"])
    assert saved["param_groups"][0]["params"] == before
    assert set(saved["state"]) == {0, 1}


def test_unnamable_param_placeholders_never_match_each_other():
    """Placeholders are POSITIONAL, so the same string means a different tensor
    in each process — and a narrowed restart's list is shorter, which lines
    ``<unnamed>.N`` up with a DIFFERENT saved param. Matching them would
    transplant a foreign param's moments (or throw a shape error at the first
    step, far from the cause)."""
    a, b, c = (torch.nn.Parameter(torch.randn(4, 4)) for _ in range(3))
    saved = _adam_state([a, b, c]).state_dict()
    # Saved position 1 is unnamable; after the narrowing, position 1 of the
    # CURRENT list is a different unnamable param.
    remapped, missing = remap_optimizer_state(
        saved,
        saved_names=["m.a", "<unnamed>.1", "m.c"],
        current_names=["m.a", "<unnamed>.1"],
    )
    assert missing == ["<unnamed>.1"]
    assert set(remapped["state"]) == {0}  # nothing was transplanted
    assert torch.equal(remapped["state"][0]["exp_avg"], saved["state"][0]["exp_avg"])

    new_opt = torch.optim.AdamW([a, b], lr=1e-3)
    new_opt.load_state_dict(remapped)
    assert 1 not in new_opt.state_dict()["state"]


def test_placeholder_in_the_saved_list_alone_is_not_matchable():
    """Excluded on BOTH sides: a saved placeholder must not be handed to a
    param that happens to carry the same positional string this run."""
    a, b = (torch.nn.Parameter(torch.randn(4, 4)) for _ in range(2))
    saved = _adam_state([a, b]).state_dict()
    remapped, missing = remap_optimizer_state(
        saved, saved_names=["<unnamed>.0", "m.b"], current_names=["<unnamed>.0", "m.b"]
    )
    assert missing == ["<unnamed>.0"]
    assert set(remapped["state"]) == {1}


def test_param_that_never_stepped_is_reported_not_invented():
    """Optimizer state is allocated lazily, so a param present in the saved
    NAME list can still have no entry. It must be reported like any other
    unmappable param instead of producing a bogus key."""
    a, b = (torch.nn.Parameter(torch.randn(4, 4)) for _ in range(2))
    opt = torch.optim.AdamW([a, b], lr=1e-3)
    a.grad = torch.ones_like(a)
    opt.step()  # only a gets state
    saved = opt.state_dict()
    remapped, missing = remap_optimizer_state(saved, ["m.a", "m.b"], ["m.b", "m.a"])
    assert missing == ["m.b"]
    assert set(remapped["state"]) == {1}
