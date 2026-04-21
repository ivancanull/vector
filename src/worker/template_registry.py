from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from worker.schemas import JobRequest, TemplateConfig, TemplatesConfig
from worker.utils import WorkerError


@dataclass(frozen=True)
class ResolvedTemplate:
    template_name: str
    project_path: Path
    design_name: str
    setup_name: str
    sweep_name: str


class TemplateRegistry:
    def __init__(self, config: TemplatesConfig):
        self._templates = config.templates

    def has_template(self, template_name: str) -> bool:
        return template_name in self._templates

    def resolve_for_job(self, job: JobRequest) -> ResolvedTemplate:
        template = self.get(job.template_name)
        project_path = template.project_path
        if not project_path.exists():
            raise WorkerError(
                "missing_template_file", f"Template file not found: {project_path}"
            )
        return ResolvedTemplate(
            template_name=job.template_name,
            project_path=project_path,
            design_name=job.design_name or template.design_name,
            setup_name=job.setup_name or template.setup_name,
            sweep_name=job.sweep_name or template.sweep_name,
        )

    def get(self, template_name: str) -> TemplateConfig:
        template = self._templates.get(template_name)
        if template is None:
            raise WorkerError("unknown_template", f"Unknown template: {template_name}")
        return template
