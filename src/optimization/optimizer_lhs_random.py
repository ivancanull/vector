from __future__ import annotations

from typing import Any

import numpy as np

from .optimizer_base import OptimizerBase, SelectionResult


class OptimizerLHSRandom(OptimizerBase):
    """Latin-hypercube random optimizer over any CandidateSpace."""

    optimizer_backend = "lhs_random"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lhs_sequence = self._build_lhs_sequence()

    def plan_next(self, iteration: int) -> list[SelectionResult]:
        return [self._select_lhs_candidate(iteration)]

    def _initial_selections(self) -> list[SelectionResult]:
        return [self._select_lhs_candidate(len(self._history))]

    def _build_lhs_sequence(self) -> np.ndarray:
        total_trials = int(self.n_init + self.budget)
        dim = len(self.candidate_space.variable_order)
        if total_trials <= 0:
            return np.empty((0, dim), dtype=float)

        rng = np.random.default_rng(self.seed)
        sequence = np.empty((total_trials, dim), dtype=float)
        for col in range(dim):
            strata = rng.permutation(total_trials)
            jitter = rng.random(total_trials)
            sequence[:, col] = (strata + jitter) / float(total_trials)
        return sequence

    def _select_lhs_candidate(self, lhs_index: int) -> SelectionResult:
        candidates = self.candidate_space.remaining()
        if len(candidates) == 0:
            raise RuntimeError("No candidates left to select")
        if len(self._lhs_sequence) == 0:
            raise RuntimeError("No LHS samples are configured")

        sequence_index = min(int(lhs_index), len(self._lhs_sequence) - 1)
        target_normalized = np.asarray(self._lhs_sequence[sequence_index], dtype=float)
        candidates_normalized = np.asarray(self.candidate_space.normalize(candidates), dtype=float)
        distances = np.linalg.norm(candidates_normalized - target_normalized.reshape(1, -1), axis=1)
        selected_pos = int(np.argmin(distances))

        return SelectionResult(
            requested_geometry=np.asarray(candidates[selected_pos], dtype=float),
            selection_debug={
                "mode": "lhs_random",
                "lhs_index": int(sequence_index),
                "candidate_pool_size": int(len(candidates)),
                "nearest_distance_normalized": float(distances[selected_pos]),
                "selected_pos": selected_pos,
            },
        )
