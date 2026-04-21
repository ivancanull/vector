from src.common.job_types import SimulationJob


class Optimizer:
    """Placeholder for candidate generation logic."""

    def propose_next_batch(self, batch_size: int = 1) -> list[SimulationJob]:
        jobs: list[SimulationJob] = []
        for idx in range(batch_size):
            jobs.append(
                SimulationJob(
                    job_id=f"job_{idx}",
                    design_type="placeholder",
                    parameters={"x": float(idx)},
                )
            )
        return jobs
