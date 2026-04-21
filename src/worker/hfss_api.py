from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

from worker.artifact_exporter import (
    build_artifact_paths,
    export_csv_from_touchstone,
    to_result_artifacts,
    verify_artifacts_exist,
)
from worker.schemas import ArtifactPaths, JobRequest, JobResult, WorkerConfig
from worker.template_registry import ResolvedTemplate
from worker.utils import WorkerError, utc_now


def _import_hfss_class():
    try:
        from pyaedt import Hfss  # type: ignore

        return Hfss
    except Exception:
        try:
            from ansys.aedt.core import Hfss  # type: ignore

            return Hfss
        except Exception as exc:
            raise WorkerError("hfss_start_failure", f"PyAEDT import failure: {exc}") from exc


def open_hfss_session(project_path: str, design_name: str, non_graphical: bool, new_desktop: bool):
    Hfss = _import_hfss_class()
    try:
        hfss = Hfss(
            project=project_path,
            design=design_name,
            non_graphical=non_graphical,
            new_desktop_session=new_desktop,
            close_on_exit=False,
        )
        return hfss
    except Exception as exc:
        raise WorkerError("hfss_start_failure", f"Failed to open HFSS session: {exc}") from exc


def apply_variables(hfss, variables: dict[str, str]) -> None:
    try:
        for name, value in variables.items():
            hfss[name] = value
    except Exception as exc:
        raise WorkerError("geometry_failure", f"Failed to set variable(s): {exc}") from exc


def run_analysis(hfss, setup_name: str) -> None:
    try:
        ok = hfss.analyze_setup(setup_name)
        if ok is False:
            raise WorkerError("mesh_failure", f"HFSS analyze_setup returned False for {setup_name}")
    except WorkerError:
        raise
    except Exception as exc:
        raise WorkerError("mesh_failure", f"Failed while solving setup {setup_name}: {exc}") from exc


def export_touchstone(hfss, setup_name: str, sweep_name: str, out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        exported = hfss.export_touchstone(
            setup_name=setup_name,
            sweep_name=sweep_name,
            file_name=str(out_path),
        )
    except TypeError:
        try:
            exported = hfss.export_touchstone(file_name=str(out_path))
        except Exception as exc:
            raise WorkerError("export_failure", f"Touchstone export failed: {exc}") from exc
    except Exception as exc:
        raise WorkerError("export_failure", f"Touchstone export failed: {exc}") from exc

    if isinstance(exported, str):
        return exported
    return str(out_path)


def close_hfss_session(hfss, close_desktop: bool) -> None:
    try:
        hfss.release_desktop(close_projects=False, close_desktop=close_desktop)
    except Exception:
        pass


def _should_export_touchstone(job: JobRequest, cfg: WorkerConfig) -> bool:
    if job.artifact_options.export_touchstone is None:
        return cfg.default_export_touchstone
    return job.artifact_options.export_touchstone


def _should_export_csv(job: JobRequest, cfg: WorkerConfig) -> bool:
    if job.artifact_options.export_csv is None:
        return cfg.default_export_csv
    return job.artifact_options.export_csv


def _execute(job: JobRequest, template_cfg: ResolvedTemplate, worker_cfg: WorkerConfig) -> ArtifactPaths:
    need_touchstone = _should_export_touchstone(job, worker_cfg)
    need_csv = _should_export_csv(job, worker_cfg)
    if job.metrics_to_extract and not need_touchstone:
        raise WorkerError(
            "invalid_job_json",
            "metrics_to_extract requires touchstone export to be enabled",
        )

    local_artifacts = build_artifact_paths(job.job_id, worker_cfg)
    hfss = None
    try:
        hfss = open_hfss_session(
            project_path=str(template_cfg.project_path),
            design_name=template_cfg.design_name,
            non_graphical=worker_cfg.non_graphical,
            new_desktop=worker_cfg.new_desktop,
        )
        apply_variables(hfss, job.variables)
        run_analysis(hfss, template_cfg.setup_name)
        exported_touchstone = export_touchstone(
            hfss,
            template_cfg.setup_name,
            template_cfg.sweep_name,
            local_artifacts.touchstone,
        )
        actual_touchstone = Path(exported_touchstone)
        if actual_touchstone.suffix.lower().startswith(".s") and actual_touchstone.suffix.lower().endswith("p"):
            local_artifacts = type(local_artifacts)(
                touchstone=actual_touchstone,
                csv=local_artifacts.csv,
                log=local_artifacts.log,
            )
        if need_csv:
            export_csv_from_touchstone(local_artifacts.touchstone, local_artifacts.csv)
        verify_artifacts_exist(local_artifacts, need_touchstone, need_csv)
        return to_result_artifacts(local_artifacts, worker_cfg.queue_root)
    finally:
        if hfss is not None:
            close_hfss_session(hfss, close_desktop=worker_cfg.close_desktop_after_job)


def execute_job(job: JobRequest, template_cfg: ResolvedTemplate, worker_cfg: WorkerConfig) -> JobResult:
    start_time = utc_now()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_execute, job, template_cfg, worker_cfg)
        try:
            artifacts = future.result(timeout=worker_cfg.job_timeout_sec)
            finish_time = utc_now()
            return JobResult(
                job_id=job.job_id,
                status="done",
                worker_name=worker_cfg.worker_name,
                start_time=start_time,
                finish_time=finish_time,
                metrics=None,
                artifacts=artifacts,
                error_message=None,
            )
        except FuturesTimeoutError as exc:
            raise WorkerError("solve_timeout", "HFSS execution timed out") from exc
