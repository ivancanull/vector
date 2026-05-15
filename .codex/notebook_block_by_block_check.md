# Notebook Block-by-Block Check Rule

After generating or modifying any Jupyter notebook, always verify it **block by block** before finalizing.

## Required checks (in order)
1. **Read each cell sequentially** and confirm logic/data flow is consistent with previous cells.
2. **Validate algorithm intent** per cell:
   - equations/metrics match objective,
   - thresholds/ranges are correct,
   - fallback paths are explicit.
3. **Validate I/O and dependencies** per cell:
   - paths exist,
   - imports are available,
   - variable names are defined before use.
4. **Run notebook top-to-bottom** and inspect outputs cell-by-cell:
   - counters/shapes/ranges are sensible,
   - no hidden unit mismatch,
   - no silent skips without reporting.
5. **Check final result cell**:
   - summary values are finite and in expected range,
   - plots match printed conclusions.
6. **If any cell fails**, patch and re-run from the top until all checks pass.

## Non-negotiable
- Do not mark notebook work as done without block-by-block review + one full execution pass.
