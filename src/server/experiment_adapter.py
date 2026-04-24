from __future__ import annotations

from common.job_types import BatchCase, BatchSimulationJob
from server.experiment_models import CaseVariant, Experiment


def to_batch_case(variant: CaseVariant) -> BatchCase:
    return BatchCase(
        case_id=variant.case_id,
        variables=variant.variables,
        output_filename=variant.output_filename,
    )


def to_batch_simulation_job(experiment: Experiment) -> BatchSimulationJob:
    experiment.validate_cases_against_design_space()
    return BatchSimulationJob(
        batch_id=experiment.experiment_id,
        template_name=experiment.template_name,
        cases=[to_batch_case(case) for case in experiment.cases],
        metrics_to_extract=experiment.metrics_to_extract(),
        design_name=experiment.design_name,
        setup_name=experiment.setup_name,
        sweep_name=experiment.sweep_name,
        artifact_options=experiment.artifact_options,
        retry_count=experiment.retry_count,
    )
