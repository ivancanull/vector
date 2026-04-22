from pathlib import Path
from types import SimpleNamespace

from scripts.worker import spiral_inductor_run_one


def test_run_one_returns_zero_when_no_pending_job(monkeypatch, tmp_path: Path) -> None:
    cfg = SimpleNamespace(
        templates_yaml=tmp_path / "templates.yaml",
        queue_root=tmp_path / "queue",
        logs_dir=tmp_path / "logs",
    )

    monkeypatch.setattr(spiral_inductor_run_one, "load_worker_config", lambda _path: cfg)
    monkeypatch.setattr(spiral_inductor_run_one, "load_templates_config", lambda _path: object())
    monkeypatch.setattr(spiral_inductor_run_one, "ensure_runtime_dirs", lambda _q, _l: None)
    monkeypatch.setattr(spiral_inductor_run_one, "TemplateRegistry", lambda _cfg: object())
    monkeypatch.setattr(spiral_inductor_run_one, "setup_global_logger", lambda _logs: object())
    monkeypatch.setattr(spiral_inductor_run_one, "try_claim_one_job", lambda _queue_root: None)

    called = {"process": False}

    def _process(*_args, **_kwargs):
        called["process"] = True

    monkeypatch.setattr(spiral_inductor_run_one, "process_claimed_job", _process)

    rc = spiral_inductor_run_one.run_one(tmp_path / "worker.yaml")
    assert rc == 0
    assert called["process"] is False


def test_run_one_processes_claimed_job(monkeypatch, tmp_path: Path) -> None:
    claimed_path = tmp_path / "queue" / "running" / "job_01.json"
    cfg = SimpleNamespace(
        templates_yaml=tmp_path / "templates.yaml",
        queue_root=tmp_path / "queue",
        logs_dir=tmp_path / "logs",
    )

    monkeypatch.setattr(spiral_inductor_run_one, "load_worker_config", lambda _path: cfg)
    monkeypatch.setattr(spiral_inductor_run_one, "load_templates_config", lambda _path: object())
    monkeypatch.setattr(spiral_inductor_run_one, "ensure_runtime_dirs", lambda _q, _l: None)

    registry = object()
    logger = object()
    monkeypatch.setattr(spiral_inductor_run_one, "TemplateRegistry", lambda _cfg: registry)
    monkeypatch.setattr(spiral_inductor_run_one, "setup_global_logger", lambda _logs: logger)
    monkeypatch.setattr(
        spiral_inductor_run_one, "try_claim_one_job", lambda _queue_root: claimed_path
    )

    captured = {}

    def _process(path, cfg_obj, registry_obj, logger_obj):
        captured["path"] = path
        captured["cfg"] = cfg_obj
        captured["registry"] = registry_obj
        captured["logger"] = logger_obj

    monkeypatch.setattr(spiral_inductor_run_one, "process_claimed_job", _process)

    rc = spiral_inductor_run_one.run_one(tmp_path / "worker.yaml")
    assert rc == 0
    assert captured["path"] == claimed_path
    assert captured["cfg"] is cfg
    assert captured["registry"] is registry
    assert captured["logger"] is logger
