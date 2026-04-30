from __future__ import annotations

from typing import Any

import numpy as np

from .optimizer_base import OptimizerBase, SelectionResult


class OptimizerRandom(OptimizerBase):
    """Random-selection optimizer over any CandidateSpace."""

    optimizer_backend = "random"

    def plan_next(self, iteration: int) -> list[SelectionResult]:
        candidates = self.candidate_space.remaining()
        if len(candidates) == 0:
            raise RuntimeError("No candidates left to select")
        pos = int(self._rng.integers(0, len(candidates)))
        return [
            SelectionResult(
                requested_geometry=np.asarray(candidates[pos], dtype=float),
                selection_debug={
                    "mode": "random",
                    "selected_pos": pos,
                    "candidate_pool_size": int(len(candidates)),
                },
            )
        ]
