from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    queue: Path
    pending: Path
    running: Path
    done: Path
    failed: Path
    results: Path
    database: Path
    experiments_db: Path



def default_paths(root: str | Path = ".") -> ProjectPaths:
    root_path = Path(root).resolve()
    queue = root_path / "queue"
    database = root_path / "database"
    return ProjectPaths(
        root=root_path,
        queue=queue,
        pending=queue / "pending",
        running=queue / "running",
        done=queue / "done",
        failed=queue / "failed",
        results=queue / "results",
        database=database,
        experiments_db=database / "experiments.db",
    )
