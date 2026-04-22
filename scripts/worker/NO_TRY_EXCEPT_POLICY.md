# No `try` / `except` in `scripts/worker`

This folder should avoid `try` / `except` blocks.

## Rule

- Do not use `try` / `except` in files under `scripts/worker/`.
- Prefer explicit checks (`hasattr`, `getattr`, version checks, return-value checks) and clear early returns.
- If an API compatibility branch is needed, use feature detection or package/version detection, not exception-driven control flow.

## Current `try` / `except` usage to remove

File: `scripts/worker/spiral_inductor_run_one.py`

1. Imports and setup
- Current: no `try` / `except` required.
- Keep explicit precondition checks and return codes.

## Migration checklist

- [x] Remove all `try` / `except` from `scripts/worker/spiral_inductor_run_one.py`.
- [x] Keep behavior equivalent for queue claim + one-job execution.
- [x] Add explicit branch logic so failure reasons remain visible.
- [x] Re-run script with representative parameters to verify output and exit codes.
