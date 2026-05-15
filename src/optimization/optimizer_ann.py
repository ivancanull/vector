from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from modeling import ResponseSurrogate, ResponseTCNConfig, ResponseTrainingSample, stack_training_samples

from .optimizer_base import OptimizerBase, SelectionResult, SimulationRecord


class OptimizerANN(OptimizerBase):
    """Online optimizer using a response ANN instead of a GP posterior."""

    optimizer_backend = "ann"

    def __init__(
        self,
        *,
        model_config: dict[str, Any] | ResponseTCNConfig | None = None,
        port_combinations: Sequence[tuple[int, int] | list[int]] = ((0, 0), (0, 1)),
        retrain_every: int = 1,
        epochs: int = 200,
        batch_size: int = 16,
        lr: float = 1e-3,
        device: str | None = None,
        use_trust_region: bool = False,
        tr_length_init: float = 0.25,
        tr_length_min: float = 0.03,
        tr_length_max: float = 1.0,
        tr_success_tolerance: int = 3,
        tr_failure_tolerance: int = 6,
        tr_expand_factor: float = 2.0,
        tr_shrink_factor: float = 0.5,
        tr_improvement_epsilon: float = 1e-9,
        **kwargs: Any,
    ) -> None:
        if retrain_every <= 0:
            raise ValueError("retrain_every must be > 0")
        if tr_length_init <= 0 or tr_length_min <= 0 or tr_length_max <= 0:
            raise ValueError("trust-region lengths must be > 0")
        if tr_length_min > tr_length_init or tr_length_init > tr_length_max:
            raise ValueError("require tr_length_min <= tr_length_init <= tr_length_max")
        if tr_success_tolerance <= 0 or tr_failure_tolerance <= 0:
            raise ValueError("trust-region success/failure tolerances must be > 0")
        if tr_expand_factor <= 1.0:
            raise ValueError("tr_expand_factor must be > 1.0")
        if not (0.0 < tr_shrink_factor < 1.0):
            raise ValueError("tr_shrink_factor must be in (0, 1)")

        self.retrain_every = int(retrain_every)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.device = device
        self.port_combinations = [tuple(map(int, combo)) for combo in port_combinations]
        if self.port_combinations != [(0, 0), (0, 1)]:
            raise ValueError("OptimizerANN currently supports port_combinations [(0, 0), (0, 1)]")

        self.use_trust_region = bool(use_trust_region)
        self.tr_length_init = float(tr_length_init)
        self.tr_length_min = float(tr_length_min)
        self.tr_length_max = float(tr_length_max)
        self.tr_success_tolerance = int(tr_success_tolerance)
        self.tr_failure_tolerance = int(tr_failure_tolerance)
        self.tr_expand_factor = float(tr_expand_factor)
        self.tr_shrink_factor = float(tr_shrink_factor)
        self.tr_improvement_epsilon = float(tr_improvement_epsilon)
        self._tr_center_norm: np.ndarray | None = None
        self._tr_length = float(tr_length_init)
        self._tr_success_count = 0
        self._tr_failure_count = 0
        self._tr_best_objective: float | None = None
        self._last_tr_update_action = "init"

        self._model_config_input = model_config
        self._model_config: ResponseTCNConfig | None = None
        self._surrogate: ResponseSurrogate | None = None
        self._model_ready = False
        self._last_fit_error: str | None = None
        self._training_samples: list[ResponseTrainingSample] = []
        self._freqs: np.ndarray | None = None
        super().__init__(**kwargs)
        self._configure_surrogate()

    @property
    def ann_training_path(self) -> Path:
        return self.checkpoint_path.with_suffix(".ann_training.npz")

    def plan_next(self, iteration: int) -> list[SelectionResult]:
        self._update_trust_region_state()
        if self.state.step % self.retrain_every == 0:
            self._fit_model()
        selection = self._select_next(self.candidate_space.remaining())
        return [selection]

    def collect_post_simulation_data(
        self,
        records: Sequence[SimulationRecord],
        *,
        iteration: int,
    ) -> list[dict[str, Any]]:
        trials = super().collect_post_simulation_data(records, iteration=iteration)
        for record, trial in zip(records, trials):
            self._maybe_add_training_sample(record, trial)
        return trials

    def resume_from_checkpoint(self, path: str | Path) -> None:
        super().resume_from_checkpoint(path)
        self._load_training_sidecar()

        try:
            import json

            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return

        ann = payload.get("ann_surrogate", {})
        if isinstance(ann, dict) and ann.get("freqs") is not None:
            self._freqs = np.asarray(ann["freqs"], dtype=float).reshape(-1)

        tr = payload.get("trust_region", {})
        if not isinstance(tr, dict):
            return
        center = tr.get("center_norm")
        self._tr_center_norm = None if center is None else np.asarray(center, dtype=float).reshape(-1)
        self._tr_length = float(tr.get("length", self._tr_length))
        self._tr_success_count = int(tr.get("success_count", 0))
        self._tr_failure_count = int(tr.get("failure_count", 0))
        best = tr.get("best_objective")
        self._tr_best_objective = None if best is None else float(best)
        self._last_tr_update_action = str(tr.get("last_action", self._last_tr_update_action))

    def _configure_surrogate(self) -> None:
        input_dim = len(self.candidate_space.variable_order)
        if isinstance(self._model_config_input, ResponseTCNConfig):
            model_config = self._model_config_input
            if model_config.input_dim != input_dim:
                raise ValueError(f"model input_dim {model_config.input_dim} does not match candidate dimension {input_dim}")
        else:
            model_config = ResponseTCNConfig.from_mapping(self._model_config_input or {}, input_dim=input_dim)
        self._model_config = model_config
        self._surrogate = ResponseSurrogate(
            model_config=model_config,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            seed=0 if self.seed is None else int(self.seed),
            device=self.device,
        )

    def _fit_model(self) -> None:
        if not self._training_samples:
            self._model_ready = False
            self._last_fit_error = None
            return
        try:
            assert self._surrogate is not None
            X, r, i = stack_training_samples(self._training_samples)
            self._surrogate.fit(X, r, i)
            self._model_ready = True
            self._last_fit_error = None
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            self._model_ready = False
            self._last_fit_error = f"{type(exc).__name__}: {exc}"

    def _select_next(self, candidates_actual: np.ndarray) -> SelectionResult:
        candidates_actual = np.asarray(candidates_actual, dtype=float)
        if len(candidates_actual) == 0:
            raise RuntimeError("No candidates left to select")
        candidates_actual, tr_debug = self._apply_trust_region_filter(candidates_actual)

        if not self._model_ready or self._surrogate is None or self._freqs is None:
            return self._deterministic_selection(
                candidates_actual,
                {
                    "mode": "deterministic_not_ready",
                    "model_fit_status": "failed" if self._last_fit_error else "not_ready",
                    "fit_error": self._last_fit_error,
                    "fallback_policy": "first_remaining_candidate",
                    **tr_debug,
                },
            )

        try:
            Xcand = self.candidate_space.normalize(candidates_actual)
            prediction = self._surrogate.predict(Xcand)
            objectives = np.asarray(
                [
                    self._predicted_objective(prediction.r[pos], prediction.i[pos])
                    for pos in range(len(candidates_actual))
                ],
                dtype=float,
            )
            finite = np.isfinite(objectives)
            if not np.any(finite):
                raise ValueError("all predicted objectives are non-finite")
            masked = np.where(finite, objectives, np.inf)
            pos = int(np.argmin(masked))
            return SelectionResult(
                requested_geometry=np.asarray(candidates_actual[pos], dtype=float),
                predicted_mean=float(objectives[pos]),
                acquisition_score=float(-objectives[pos]),
                selection_debug={
                    "mode": "ann_predicted_objective",
                    "model_fit_status": "ok",
                    "selected_pos": pos,
                    "candidate_pool_size": int(len(candidates_actual)),
                    "predicted_objective_min": float(objectives[pos]),
                    **tr_debug,
                    "ann_runtime": {
                        "x_is_normalized": True,
                        "num_training_samples": int(len(self._training_samples)),
                    },
                },
            )
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            return self._deterministic_selection(
                candidates_actual,
                {
                    "mode": "deterministic_select_error",
                    "model_fit_status": "failed",
                    "fit_error": self._last_fit_error,
                    "select_error": f"{type(exc).__name__}: {exc}",
                    "fallback_policy": "first_remaining_candidate",
                    **tr_debug,
                },
            )

    def _predicted_objective(self, r: np.ndarray, i: np.ndarray) -> float:
        assert self._freqs is not None
        s_params = self._response_arrays_to_sparams(r, i)
        result = self.optimization_spec.evaluate(self._freqs, s_params)
        return result.scalar_objective()

    def _response_arrays_to_sparams(self, r: np.ndarray, i: np.ndarray) -> np.ndarray:
        assert self._model_config is not None
        response = np.asarray(r, dtype=float) + 1j * np.asarray(i, dtype=float)
        if response.shape != self._model_config.output_channels["r"]:
            raise ValueError(f"predicted response shape mismatch: got {response.shape}")
        length = response.shape[1]
        s_params = np.zeros((2, 2, length), dtype=complex)
        for channel_idx, (row, col) in enumerate(self.port_combinations):
            s_params[row, col, :] = response[channel_idx]
            if (row, col) == (0, 1):
                # Existing objectives read S21 as (1, 0); fill reciprocal channel for 2-port filters.
                s_params[1, 0, :] = response[channel_idx]
        return s_params

    def _maybe_add_training_sample(self, record: SimulationRecord, trial: dict[str, Any]) -> None:
        simulation = record.simulation
        if simulation.status != "done" or simulation.freqs is None or simulation.s_params is None:
            return
        assert self._model_config is not None
        freqs = np.asarray(simulation.freqs, dtype=float).reshape(-1)
        if freqs.shape[0] != self._model_config.output_length:
            raise ValueError(f"simulation frequency length {freqs.shape[0]} does not match ANN output length {self._model_config.output_length}")
        if self._freqs is None:
            self._freqs = freqs
        elif not np.allclose(freqs, self._freqs):
            raise ValueError("simulation frequency grid changed during ANN optimizer run")

        s_params = np.asarray(simulation.s_params, dtype=complex)
        if s_params.ndim != 3 or s_params.shape[2] != self._model_config.output_length:
            raise ValueError(f"simulation s_params shape {s_params.shape} is incompatible with ANN output")
        selected = np.stack([s_params[row, col, :] for row, col in self.port_combinations], axis=0)
        self._training_samples.append(
            ResponseTrainingSample(
                geometry_normalized=np.asarray(trial["geometry_normalized"], dtype=float).reshape(-1),
                r=np.real(selected),
                i=np.imag(selected),
            )
        )

    def _deterministic_selection(self, candidates_actual: np.ndarray, debug: dict[str, Any]) -> SelectionResult:
        return SelectionResult(
            requested_geometry=np.asarray(candidates_actual[0], dtype=float),
            selection_debug=debug,
        )

    def _apply_trust_region_filter(self, candidates_actual: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if not self.use_trust_region:
            return candidates_actual, {"tr_active": False}
        center = self._resolve_tr_center_norm()
        if center is None:
            return candidates_actual, {
                "tr_active": True,
                "tr_filter_applied": False,
                "tr_filter_reason": "center_unavailable",
                "tr_length": float(self._tr_length),
            }

        Xcand_np = self.candidate_space.normalize(candidates_actual)
        d_inf = np.max(np.abs(Xcand_np - center.reshape(1, -1)), axis=1)
        inside = d_inf <= float(self._tr_length)
        tr_count = int(np.count_nonzero(inside))
        if tr_count == 0:
            return candidates_actual, {
                "tr_active": True,
                "tr_filter_applied": False,
                "tr_filter_reason": "empty_region_fallback_global",
                "tr_length": float(self._tr_length),
                "tr_candidate_count": int(len(candidates_actual)),
                "tr_candidates_in_region": 0,
                "tr_center": center.tolist(),
                "tr_update_action": self._last_tr_update_action,
            }
        filtered = candidates_actual[inside]
        return filtered, {
            "tr_active": True,
            "tr_filter_applied": True,
            "tr_length": float(self._tr_length),
            "tr_candidate_count": int(len(candidates_actual)),
            "tr_candidates_in_region": tr_count,
            "tr_center": center.tolist(),
            "tr_update_action": self._last_tr_update_action,
        }

    def _resolve_tr_center_norm(self) -> np.ndarray | None:
        if self._tr_center_norm is not None:
            return np.asarray(self._tr_center_norm, dtype=float).reshape(-1)
        if self.state.best_geometry is None:
            return None
        self._tr_center_norm = np.asarray(
            self.candidate_space.normalize(np.asarray(self.state.best_geometry, dtype=float)),
            dtype=float,
        ).reshape(-1)
        return np.asarray(self._tr_center_norm, dtype=float).reshape(-1)

    def _update_trust_region_state(self) -> None:
        if not self.use_trust_region:
            return
        if self.state.best_geometry is not None:
            self._tr_center_norm = np.asarray(
                self.candidate_space.normalize(np.asarray(self.state.best_geometry, dtype=float)),
                dtype=float,
            ).reshape(-1)
        current_best = self.state.best_objective
        if current_best is None:
            self._last_tr_update_action = "wait_for_best"
            return
        if self._tr_best_objective is None:
            self._tr_best_objective = float(current_best)
            self._tr_success_count = 0
            self._tr_failure_count = 0
            self._last_tr_update_action = "init_best"
            return

        improved = float(current_best) < float(self._tr_best_objective) - self.tr_improvement_epsilon
        if improved:
            self._tr_best_objective = float(current_best)
            self._tr_success_count += 1
            self._tr_failure_count = 0
            if self._tr_success_count >= self.tr_success_tolerance:
                self._tr_length = min(self.tr_length_max, self._tr_length * self.tr_expand_factor)
                self._tr_success_count = 0
                self._last_tr_update_action = "expand"
            else:
                self._last_tr_update_action = "success_hold"
            return

        self._tr_failure_count += 1
        self._tr_success_count = 0
        if self._tr_failure_count >= self.tr_failure_tolerance:
            new_len = self._tr_length * self.tr_shrink_factor
            self._tr_length = max(self.tr_length_min, new_len)
            self._last_tr_update_action = "restart_at_min" if new_len < self.tr_length_min else "shrink"
            self._tr_failure_count = 0
        else:
            self._last_tr_update_action = "failure_hold"

    def _save_checkpoint(self) -> None:
        super()._save_checkpoint()
        self._save_training_sidecar()
        try:
            import json

            payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            return
        payload["ann_surrogate"] = {
            "training_sidecar": str(self.ann_training_path),
            "num_training_samples": int(len(self._training_samples)),
            "freqs": None if self._freqs is None else self._freqs.tolist(),
            "model_config": self._model_config_summary(),
        }
        payload["trust_region"] = {
            "enabled": self.use_trust_region,
            "center_norm": None if self._tr_center_norm is None else np.asarray(self._tr_center_norm, dtype=float).reshape(-1).tolist(),
            "length": float(self._tr_length),
            "success_count": int(self._tr_success_count),
            "failure_count": int(self._tr_failure_count),
            "best_objective": self._tr_best_objective,
            "last_action": self._last_tr_update_action,
        }
        self.checkpoint_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def _save_training_sidecar(self) -> None:
        self.ann_training_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._training_samples:
            assert self._model_config is not None
            channels, length = self._model_config.output_channels["r"]
            np.savez(
                self.ann_training_path,
                X=np.empty((0, len(self.candidate_space.variable_order)), dtype=float),
                r=np.empty((0, channels, length), dtype=float),
                i=np.empty((0, channels, length), dtype=float),
            )
            return
        X, r, i = stack_training_samples(self._training_samples)
        np.savez(self.ann_training_path, X=X, r=r, i=i)

    def _load_training_sidecar(self) -> None:
        path = self.ann_training_path
        if not path.exists():
            return
        data = np.load(path)
        X = np.asarray(data["X"], dtype=float)
        r = np.asarray(data["r"], dtype=float)
        i = np.asarray(data["i"], dtype=float)
        self._training_samples = [
            ResponseTrainingSample(geometry_normalized=X[pos], r=r[pos], i=i[pos])
            for pos in range(X.shape[0])
        ]

    def _model_config_summary(self) -> dict[str, Any]:
        assert self._model_config is not None
        return {
            "input_dim": self._model_config.input_dim,
            "hidden_dims": list(self._model_config.hidden_dims),
            "output_channels": {k: list(v) for k, v in self._model_config.output_channels.items()},
            "tcns": {
                "kernel_size": list(self._model_config.kernel_size),
                "stride": list(self._model_config.stride),
                "padding": list(self._model_config.padding),
                "dilation": list(self._model_config.dilation),
                "output_padding": list(self._model_config.output_padding),
            },
            "dropout": self._model_config.dropout,
        }


OnlineANNOptimizer = OptimizerANN
