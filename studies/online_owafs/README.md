# Online LHS-Database Warmup BCLF Experiments

Use one Python runner for BO, LHS, Turbo, BNN, and OWAFS-BNN online experiments.
The runner defaults to a namespace that names the actual policy: BNN-style
methods warm up from the reusable previous-LHS database, while BO/Turbo/LHS
start online directly.

```text
experiments/online_lhsdb_warmup/
```

## Study Layout

- Scripts: `studies/online_owafs/scripts`
- Data: `experiments/online_lhsdb_warmup`
- Diagnostic CSVs: `experiments/online_lhsdb_warmup/owafs_bnn/diagnostics`
- Figures: `studies/online_owafs/figures`
- Reports: `studies/online_owafs/reports`

Run commands from the repo root with the activated `vector` environment:

```bash
cd /home/zhanghf/projects/vector
conda activate vector
python studies/online_owafs/scripts/run_online_lhsdb_warmup.py --methods METHOD
```

`METHOD` can be `bo`, `lhs`, `turbo`, `bnn`, or `owafs_bnn`. Multiple methods
can be passed as separate tokens or comma-separated groups.

## Initialization Policy

The default total campaign length is 80 budgeted trials per seed:

```text
--experiment-name online_lhsdb_warmup
--total-trials 80
--output-dir experiments/online_lhsdb_warmup
```

Method-specific initialization:

```text
bo          n_init=0, first online point is random when the BO model is not ready
turbo       n_init=0, same as BO with trust-region selection after model fit
lhs         n_init=0, LHS/random baseline for the full total_trials
bnn         n_init=warmup_size, default 15 rows from the reusable LHS database
owafs_bnn   n_init=warmup_size, default 15 rows from the reusable LHS database
```

BNN warmup manifests are written only for methods that need them:

```text
experiments/online_lhsdb_warmup/initial_designs/seed*.json
```

There are only two warmup sources:

```text
--initial-design-source random
--initial-design-source previous-lhs
--lhs-dataset-dir data/bandpass_coupled_line_filter_export/ext/lhs
```

`previous-lhs` samples rows without replacement per target seed from
`lhs_online_general_rows.csv`, filtered to cache-resolved, in-bounds,
finite-objective geometries. Different target seeds may naturally sample the
same source row. This is the default, so the warmup source objective is known
before launching BNN training.

`random` generates fresh random geometries inside the configured bounds. These
points are not required to exist in the cache; server backend runs will submit
the warmup simulations before the initial 800-epoch BNN training. Because these
are new points, the preview table does not have source objectives for them.

Before launching a real BNN or OWAFS-BNN campaign, generate and inspect the
five seed warmup manifests:

```bash
python studies/online_owafs/scripts/run_online_lhsdb_warmup.py \
  --methods bnn owafs_bnn \
  --generate-initial-design-only
```

The runner writes a seed-level summary and full point table:

```text
experiments/online_lhsdb_warmup/initial_designs/warmup_initial_summary.csv
experiments/online_lhsdb_warmup/initial_designs/warmup_initial_points.csv
```

If the previous-LHS random warmup looks unfair, rebuild only the manifests:

```bash
python studies/online_owafs/scripts/run_online_lhsdb_warmup.py \
  --methods bnn owafs_bnn \
  --generate-initial-design-only \
  --rebuild-initial-designs \
  --initial-design-seed-offset 100
```

Change `--initial-design-seed-offset` again, for example to `200`, to inspect
another independent draw.

To inspect the fresh random source instead of previous-LHS:

```bash
python studies/online_owafs/scripts/run_online_lhsdb_warmup.py \
  --methods bnn owafs_bnn \
  --initial-design-source random \
  --generate-initial-design-only
```

If active manifests under the output root use a different policy, the runner
archives active method outputs, `initial_designs/`, and local BNN checkpoints to:

```text
experiments/online_lhsdb_warmup/archive/<UTC>_pre_<new_policy>/
```

## BNN Training Schedule

BNN and OWAFS-BNN force `retrain_every=1`.

The initial warmup training uses:

```text
initial_epochs = 800
```

After warmup, online update `k` trains on the full current history for:

```text
20 * k epochs
```

So the first two online updates use 20 and 40 epochs. The schedule is recorded
in BNN checkpoint metadata as `linear_online_iteration`.

## Common Commands

Run BO with no warmup:

```bash
python studies/online_owafs/scripts/run_online_lhsdb_warmup.py --methods bo
```

Run BNN with default 15-row LHS-database warmup:

```bash
python studies/online_owafs/scripts/run_online_lhsdb_warmup.py --methods bnn
```

Run OWAFS-BNN:

```bash
python studies/online_owafs/scripts/run_online_lhsdb_warmup.py --methods owafs_bnn
```

Generate only BNN/OWAFS-BNN warmup manifests:

```bash
python studies/online_owafs/scripts/run_online_lhsdb_warmup.py \
  --methods bnn owafs_bnn \
  --generate-initial-design-only
```

Dry-run the full plan:

```bash
python studies/online_owafs/scripts/run_online_lhsdb_warmup.py --dry-run
```

Restart one method/seed:

```bash
python studies/online_owafs/scripts/run_online_lhsdb_warmup.py \
  --methods bo \
  --seeds 0 \
  --reset-checkpoint
```

Inspect all options:

```bash
python studies/online_owafs/scripts/run_online_lhsdb_warmup.py --help
```

## Outputs

Each method writes into its own subdirectory:

```text
experiments/online_lhsdb_warmup/bo/
experiments/online_lhsdb_warmup/lhs/
experiments/online_lhsdb_warmup/turbo/
experiments/online_lhsdb_warmup/bnn/
experiments/online_lhsdb_warmup/owafs_bnn/
```

Run ids and files use:

```text
{method}_{experiment_name}_seed{seed}
```

For example:

```text
bnn_online_lhsdb_warmup_seed0
```

BNN checkpoints live in:

```text
experiments/online_lhsdb_warmup/bnn_checkpoints/
```

## Python Environment

Use `python`, not `python3`, from the activated `vector` conda environment. On
this machine, CAD tools can put a Calibre `python3` wrapper earlier in `PATH`,
which is not suitable for these experiments.

If activation is unclear, use the environment Python explicitly:

```bash
/home/zhanghf/anaconda3/envs/vector/bin/python \
  studies/online_owafs/scripts/run_online_lhsdb_warmup.py --methods owafs_bnn
```
