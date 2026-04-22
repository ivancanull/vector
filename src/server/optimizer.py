from common.job_types import SimulationJob


class Optimizer:
    """Placeholder for candidate generation logic."""

    def propose_next_batch(self, batch_size: int = 1) -> list[SimulationJob]:
        jobs: list[SimulationJob] = []
        for idx in range(batch_size):
            jobs.append(
                SimulationJob(
                    job_id=f"job_{idx}",
                    template_name="si_template_v1",
                    variables={"x": f"{float(idx)}mm"},
                    output_filename=f"job_{idx}.s2p",
                    metrics_to_extract=["max_s11_db"],
                )
            )
        return jobs
