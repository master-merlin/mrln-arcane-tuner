import pytest

from app.core.self_update import SelfUpdateService, UpdateState


@pytest.fixture
def svc():
    return SelfUpdateService(app_dir="/fake/app", branch="main", remote="https://example/repo.git")


def test_initial_state_is_idle(svc):
    assert svc.state == UpdateState.IDLE


def test_git_status_parses_branch_commit_dirty(svc, monkeypatch):
    def fake_run(args, cwd=None, timeout=None):
        if args[:2] == ["rev-parse", "--abbrev-ref"]:
            return (0, "main\n", "")
        if args[:2] == ["rev-parse", "--short"]:
            return (0, "a1b3c2d\n", "")
        if args[0] == "status":
            return (0, " M backend/app/x.py\n", "")
        return (0, "", "")

    monkeypatch.setattr(svc, "_run_git", fake_run)
    st = svc.git_status()
    assert st["branch"] == "main"
    assert st["commit"] == "a1b3c2d"
    assert st["dirty"] is True
    assert st["is_repo"] is True


def test_git_status_not_a_repo(svc, monkeypatch):
    monkeypatch.setattr(svc, "_run_git", lambda *a, **k: (128, "", "not a git repository"))
    st = svc.git_status()
    assert st["is_repo"] is False


def test_probe_availability_true_when_lsremote_ok(svc, monkeypatch):
    monkeypatch.setattr(svc, "_run_git", lambda *a, **k: (0, "ref\tHEAD\n", ""))
    svc.probe_availability()
    assert svc.available is True


def test_probe_availability_false_when_lsremote_fails(svc, monkeypatch):
    monkeypatch.setattr(svc, "_run_git", lambda *a, **k: (128, "", "Authentication failed"))
    svc.probe_availability()
    assert svc.available is False


def test_active_task_count_counts_running_tasks_only(svc, monkeypatch):
    from app.core import self_update as mod
    from app.core.tasks.task import TaskStatus

    class _Task:
        def __init__(self, status):
            self.status = status

    monkeypatch.setattr(mod, "_list_tasks",
                        lambda: [_Task(TaskStatus.RUNNING), _Task(TaskStatus.PENDING),
                                 _Task(TaskStatus.COMPLETED), _Task(TaskStatus.RUNNING)])
    assert svc.active_task_count() == 2
