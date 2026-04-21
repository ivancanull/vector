from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class ServerConfig:
    queue_root: str = "queue"
    database_path: str = "database/experiments.db"
    poll_seconds: float = 2.0
    max_pending_jobs: int = 100



def load_server_config(path: str | Path = "configs/server.yaml") -> ServerConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return ServerConfig(**data)
