from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from worker.artifact_exporter import (
    LocalArtifactPaths,
    build_artifact_paths_with_name,
    export_csv_from_touchstone,
    to_result_artifacts,
    verify_artifacts_exist,
)
from worker.setup_designer import HfssSetupDesigner
from worker.schemas import ArtifactPaths, JobRequest, JobResult, WorkerConfig
from worker.template_registry import ResolvedTemplate
from worker.utils import WorkerError, utc_now


class HfssSession:
    def __init__(self, hfss) -> None:
        self.hfss = hfss

    def apply_variables(self, variables: dict[str, str]) -> None:
        for name, value in variables.items():
            self.hfss[name] = value

    def analyze(self, setup_name: str) -> None:
        try:
            ok = self.hfss.get_setup(setup_name).analyze()
        except Exception as exc:
            raise WorkerError("mesh_failure", f"Failed while solving setup {setup_name}: {exc}") from exc
        if ok is False:
            raise WorkerError("mesh_failure", f"HFSS analyze returned False for {setup_name}")

    def export_touchstone(self, setup_name: str, sweep_name: str, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            exported = self.hfss.export_touchstone(
                setup=setup_name,
                sweep=sweep_name,
                output_file=str(out_path),
            )
        except Exception as exc:
            raise WorkerError("export_failure", f"Touchstone export failed: {exc}") from exc
        return Path(exported) if isinstance(exported, str) else out_path

    def close(self, close_desktop: bool) -> None:
        try:
            self.hfss.release_desktop(close_projects=False, close_desktop=close_desktop)
        except Exception:
            pass


class HfssSessionFactory:
    def import_hfss_class(self):
        try:
            return import_module("pyaedt").Hfss  # type: ignore[attr-defined]
        except ModuleNotFoundError:
            try:
                return import_module("ansys.aedt.core").Hfss  # type: ignore[attr-defined]
            except ModuleNotFoundError as exc:
                raise WorkerError("hfss_start_failure", f"PyAEDT import failure: {exc}") from exc

    def open(self, template_cfg: ResolvedTemplate, worker_cfg: WorkerConfig) -> HfssSession:
        Hfss = self.import_hfss_class()

        try:
            return HfssSession(
                self._open_with_desktop_kwarg(
                    Hfss,
                    project_path=str(template_cfg.project_path),
                    design_name=template_cfg.design_name,
                    non_graphical=worker_cfg.non_graphical,
                    new_desktop=worker_cfg.new_desktop,
                    use_new_desktop_session=True,
                )
            )
        except RuntimeError as exc:
            if "Project is locked" not in str(exc):
                raise WorkerError("hfss_start_failure", f"Failed to open HFSS session: {exc}") from exc
            try:
                return HfssSession(
                    self._open_with_desktop_kwarg(
                        Hfss,
                        project_path=str(template_cfg.project_path),
                        design_name=template_cfg.design_name,
                        non_graphical=worker_cfg.non_graphical,
                        new_desktop=worker_cfg.new_desktop,
                        use_new_desktop_session=True,
                        remove_lock=True,
                    )
                )
            except Exception as retry_exc:
                raise WorkerError(
                    "hfss_start_failure", f"Failed to open HFSS session: {retry_exc}"
                ) from retry_exc
        except Exception as exc:
            raise WorkerError("hfss_start_failure", f"Failed to open HFSS session: {exc}") from exc

    def _open_with_desktop_kwarg(
        self,
        Hfss,
        *,
        project_path: str,
        design_name: str,
        non_graphical: bool,
        new_desktop: bool,
        use_new_desktop_session: bool,
        **kwargs,
    ):
        desktop_kwarg = (
            {"new_desktop_session": new_desktop}
            if use_new_desktop_session
            else {"new_desktop": new_desktop}
        )
        try:
            return Hfss(
                project=project_path,
                design=design_name,
                non_graphical=non_graphical,
                **desktop_kwarg,
                **kwargs,
            )
        except TypeError:
            if use_new_desktop_session:
                return self._open_with_desktop_kwarg(
                    Hfss,
                    project_path=project_path,
                    design_name=design_name,
                    non_graphical=non_graphical,
                    new_desktop=new_desktop,
                    use_new_desktop_session=False,
                    **kwargs,
                )
            raise

@dataclass(frozen=True)
class HfssArtifactPlan:
    local_artifacts: LocalArtifactPaths
    need_touchstone: bool
    need_csv: bool


class HfssArtifactPlanner:
    def plan(self, job: JobRequest, worker_cfg: WorkerConfig) -> HfssArtifactPlan:
        need_touchstone = self.should_export_touchstone(job, worker_cfg)
        need_csv = self.should_export_csv(job, worker_cfg)
        if job.metrics_to_extract and not need_touchstone:
            raise WorkerError(
                "invalid_job_json",
                "metrics_to_extract requires touchstone export to be enabled",
            )
        return HfssArtifactPlan(
            local_artifacts=build_artifact_paths_with_name(
                job.job_id,
                worker_cfg,
                output_filename=job.output_filename,
            ),
            need_touchstone=need_touchstone,
            need_csv=need_csv,
        )

    def should_export_touchstone(self, job: JobRequest, cfg: WorkerConfig) -> bool:
        if job.artifact_options.export_touchstone is None:
            return cfg.default_export_touchstone
        return job.artifact_options.export_touchstone

    def should_export_csv(self, job: JobRequest, cfg: WorkerConfig) -> bool:
        if job.artifact_options.export_csv is None:
            return cfg.default_export_csv
        return job.artifact_options.export_csv

    def with_exported_touchstone(
        self,
        local_artifacts: LocalArtifactPaths,
        exported_touchstone: Path,
    ) -> LocalArtifactPaths:
        if (
            exported_touchstone.suffix.lower().startswith(".s")
            and exported_touchstone.suffix.lower().endswith("p")
        ):
            local_artifacts = type(local_artifacts)(
                touchstone=exported_touchstone,
                csv=local_artifacts.csv,
                log=local_artifacts.log,
            )
        return local_artifacts

    def export_csv_if_needed(self, local_artifacts: LocalArtifactPaths, need_csv: bool) -> None:
        if need_csv:
            export_csv_from_touchstone(local_artifacts.touchstone, local_artifacts.csv)

    def verify_and_relativize(
        self,
        local_artifacts: LocalArtifactPaths,
        worker_cfg: WorkerConfig,
        *,
        need_touchstone: bool,
        need_csv: bool,
    ) -> ArtifactPaths:
        verify_artifacts_exist(local_artifacts, need_touchstone, need_csv)
        return to_result_artifacts(local_artifacts, worker_cfg.queue_root)


class HfssJobExecutor:
    def __init__(
        self,
        *,
        session_factory: HfssSessionFactory,
        setup_designer: HfssSetupDesigner,
        artifact_planner: HfssArtifactPlanner,
    ) -> None:
        self.session_factory = session_factory
        self.setup_designer = setup_designer
        self.artifact_planner = artifact_planner

    @classmethod
    def default(cls) -> "HfssJobExecutor":
        return cls(
            session_factory=HfssSessionFactory(),
            setup_designer=HfssSetupDesigner(),
            artifact_planner=HfssArtifactPlanner(),
        )

    def execute(
        self,
        job: JobRequest,
        template_cfg: ResolvedTemplate,
        worker_cfg: WorkerConfig,
    ) -> JobResult:
        start_time = utc_now()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._execute_artifacts, job, template_cfg, worker_cfg)
            try:
                artifacts = future.result(timeout=worker_cfg.job_timeout_sec)
            except FuturesTimeoutError as exc:
                raise WorkerError("solve_timeout", "HFSS execution timed out") from exc
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

    def _execute_artifacts(
        self,
        job: JobRequest,
        template_cfg: ResolvedTemplate,
        worker_cfg: WorkerConfig,
    ) -> ArtifactPaths:
        plan = self.artifact_planner.plan(job, worker_cfg)
        session: HfssSession | None = None
        try:
            session = self.session_factory.open(template_cfg, worker_cfg)
            session.apply_variables(job.variables)
            resolved_setup = self.setup_designer.ensure_setup(
                session.hfss,
                setup_name=template_cfg.setup_name,
                sweep_name=template_cfg.sweep_name,
                setup_config=template_cfg.hfss_setup,
                sweep_config=template_cfg.hfss_sweep,
            )
            session.analyze(resolved_setup.setup_name)
            exported_touchstone = session.export_touchstone(
                resolved_setup.setup_name,
                resolved_setup.sweep_name,
                plan.local_artifacts.touchstone,
            )
            local_artifacts = self.artifact_planner.with_exported_touchstone(
                plan.local_artifacts,
                exported_touchstone,
            )
            self.artifact_planner.export_csv_if_needed(local_artifacts, plan.need_csv)
            return self.artifact_planner.verify_and_relativize(
                local_artifacts,
                worker_cfg,
                need_touchstone=plan.need_touchstone,
                need_csv=plan.need_csv,
            )
        finally:
            if session is not None:
                session.close(close_desktop=worker_cfg.close_desktop_after_job)
