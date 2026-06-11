from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re
import shutil
import time

from worker.schemas import WorkerConfig
from worker.template_registry import ResolvedTemplate


TRASH_DIR_NAME = "_trash"
TRASH_TTL_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class PreparedTemplateProject:
    template: ResolvedTemplate
    work_dir: Path | None
    preserve: bool

    def cleanup(self) -> None:
        if self.work_dir is None or self.preserve:
            return
        _remove_or_trash(self.work_dir)


class TemplateProjectCopyManager:
    def prepare(
        self,
        template: ResolvedTemplate,
        *,
        worker_cfg: WorkerConfig,
        job_id: str,
    ) -> PreparedTemplateProject:
        if not worker_cfg.copy_template_per_job:
            return PreparedTemplateProject(
                template=template,
                work_dir=None,
                preserve=True,
            )

        copy_root = worker_cfg.template_copy_root or worker_cfg.queue_root / "aedt_copies"
        copy_root.mkdir(parents=True, exist_ok=True)
        cleanup_stale_trash(copy_root)

        job_component = _safe_component(job_id)
        work_dir = copy_root / job_component
        if work_dir.exists():
            _remove_or_trash(work_dir)
        if work_dir.exists():
            work_dir = _unique_work_dir(copy_root, job_component)
        work_dir.mkdir(parents=True, exist_ok=True)

        copied_project = work_dir / template.project_path.name
        shutil.copy2(template.project_path, copied_project)
        copied_template = ResolvedTemplate(
            template_name=template.template_name,
            project_path=copied_project,
            design_name=template.design_name,
            setup_name=template.setup_name,
            sweep_name=template.sweep_name,
            hfss_sweep=template.hfss_sweep,
        )
        return PreparedTemplateProject(
            template=copied_template,
            work_dir=work_dir,
            preserve=worker_cfg.preserve_local_project_copy,
        )


def cleanup_stale_trash(copy_root: Path, *, now: float | None = None) -> None:
    trash_root = copy_root / TRASH_DIR_NAME
    if not trash_root.exists():
        return
    current = time.time() if now is None else now
    for child in trash_root.iterdir():
        try:
            if current - child.stat().st_mtime >= TRASH_TTL_SECONDS:
                _remove_path(child)
        except OSError:
            pass


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "job"


def _unique_work_dir(copy_root: Path, job_component: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for index in range(1, 1000):
        candidate = copy_root / f"{job_component}_{stamp}_{index:03d}"
        if not candidate.exists():
            return candidate
    return copy_root / f"{job_component}_{stamp}_{time.time_ns()}"


def _remove_or_trash(path: Path) -> None:
    try:
        _remove_path(path)
        return
    except OSError:
        pass

    copy_root = path.parent
    trash_root = copy_root / TRASH_DIR_NAME
    trash_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = trash_root / f"{stamp}_{path.name}"
    try:
        path.replace(target)
    except OSError:
        pass


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
