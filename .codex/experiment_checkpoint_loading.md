# Experiment Checkpoint Loading Pattern

Use this pattern when adding or updating experiment notebooks that train models
for multiple seeds, train sizes, iterations, or surrogate variants. The reference
implementation is `notebooks/gp_vs_bnn_response_shape_metric.ipynb`.

## Required Controls

Each experiment should expose these controls near the main configuration cell:

```python
CHECKPOINT_DIR = OUTPUT_DIR / f"checkpoints_{SURROGATE_VARIANT}"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

USE_CHECKPOINTS = True
FORCE_RETRAIN = False
CONTINUE_LEARNING = True
UPDATE_EPOCHS_FULL = 200
UPDATE_EPOCHS_SMOKE = 1
```

- `USE_CHECKPOINTS=True`: load/save checkpoints for expensive model training.
- `FORCE_RETRAIN=True`: ignore existing checkpoints and fit from scratch.
- `CONTINUE_LEARNING=True`: when an exact checkpoint is missing, load the closest
  smaller checkpoint for the same experiment key and call `update(...)`.
- Keep checkpoint directories separated by incompatible variants, for example
  `bnn_checkpoints_raw` and `bnn_checkpoints_fourier`.

## Naming and Lookup

Use deterministic checkpoint paths that encode every compatibility-changing key:

```python
def checkpoint_path(seed: int, train_size: int) -> Path:
    return CHECKPOINT_DIR / f"{SURROGATE_VARIANT}_model_seed{int(seed)}_n{int(train_size)}.pt"
```

For online-learning experiments, include iteration or sample count instead of
`train_size`:

```python
def checkpoint_path(seed: int, iteration: int) -> Path:
    return CHECKPOINT_DIR / f"{SURROGATE_VARIANT}_model_seed{int(seed)}_iter{int(iteration):03d}.pt"
```

When continuing learning, only search checkpoints from the same seed, variant,
dataset, target mode, and experiment configuration:

```python
def find_previous_checkpoint(seed: int, train_size: int) -> tuple[int, Path] | None:
    previous_sizes = [int(size) for size in sorted(TRAIN_SIZES) if int(size) < int(train_size)]
    for previous_size in reversed(previous_sizes):
        path = checkpoint_path(seed, previous_size)
        if path.exists():
            return previous_size, path
    return None
```

## Train-or-Load Flow

Use this exact precedence:

1. If `USE_CHECKPOINTS` and exact checkpoint exists and not `FORCE_RETRAIN`, load it.
2. Else if `USE_CHECKPOINTS` and `CONTINUE_LEARNING` and not `FORCE_RETRAIN`,
   load the closest compatible previous checkpoint and call `model.update(...)`.
3. Else fit from scratch.
4. Save the resulting model to the exact checkpoint path.

Template:

```python
def train_or_load_model(seed: int, train_size: int, train_idx: np.ndarray):
    path = checkpoint_path(seed, train_size)
    device = MODEL_KWARGS.get("device")
    start = time.perf_counter()

    if USE_CHECKPOINTS and path.exists() and not FORCE_RETRAIN:
        model = MODEL_CLASS.load(path, device=device)
        return model, time.perf_counter() - start, {
            "fit_mode": "loaded_checkpoint",
            "checkpoint_path": str(path),
            "source_checkpoint_path": str(path),
            "source_train_size": int(train_size),
            "update_epochs": 0,
            "trained_epochs": int(model.trained_epochs),
        }

    previous = (
        find_previous_checkpoint(seed, train_size)
        if USE_CHECKPOINTS and CONTINUE_LEARNING and not FORCE_RETRAIN
        else None
    )
    if previous is not None:
        previous_size, previous_path = previous
        model = MODEL_CLASS.load(previous_path, device=device)
        model.update(dataset, train_idx, epochs=int(UPDATE_EPOCHS))
        fit_mode = "continued_from_checkpoint"
        source_checkpoint_path = str(previous_path)
        source_train_size = int(previous_size)
        update_epochs = int(UPDATE_EPOCHS)
    else:
        model = MODEL_CLASS(seed=int(seed), **MODEL_KWARGS)
        model.fit(dataset, train_idx)
        fit_mode = "fit_from_scratch"
        source_checkpoint_path = None
        source_train_size = None
        update_epochs = 0

    if USE_CHECKPOINTS:
        model.save(
            path,
            metadata={
                "seed": int(seed),
                "train_size": int(train_size),
                "train_indices": np.asarray(train_idx, dtype=int).tolist(),
                "fit_mode": fit_mode,
                "source_checkpoint_path": source_checkpoint_path,
                "source_train_size": source_train_size,
                "update_epochs": update_epochs,
                "experiment": EXPERIMENT_NAME,
                "surrogate_variant": SURROGATE_VARIANT,
            },
        )

    return model, time.perf_counter() - start, {
        "fit_mode": fit_mode,
        "checkpoint_path": str(path) if USE_CHECKPOINTS else None,
        "source_checkpoint_path": source_checkpoint_path,
        "source_train_size": source_train_size,
        "update_epochs": int(update_epochs),
        "trained_epochs": int(model.trained_epochs),
    }
```

## Reporting Requirements

Add checkpoint provenance to per-trial rows and summaries:

- `fit_mode`
- `checkpoint_path`
- `source_checkpoint_path`
- `source_train_size` or `source_iteration`
- `update_epochs`
- `trained_epochs`

This makes it clear whether a result came from a fresh fit, an exact checkpoint,
or continued learning.

## Compatibility Rules

- Do not share checkpoints across incompatible variants such as raw-frequency BNN
  and Fourier-frequency BNN.
- Do not reuse checkpoints when target mode, dataset, response metric, frequency
  grid, model architecture, or train split semantics change.
- Store enough metadata in `model.save(..., metadata=...)` to audit the checkpoint.
- Keep smoke mode cheap: use tiny models and `UPDATE_EPOCHS_SMOKE=1`.
- Full mode can use larger `UPDATE_EPOCHS_FULL`, matching the experiment's cost.
