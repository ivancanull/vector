# Online BNN Study

BNN online optimization runner and launch scripts.

- Scripts: `studies/online_bnn/scripts`
- Data: `experiments/bnn_online_general`
- Checkpoints: `experiments/bnn_online_general/bnn_checkpoints`
- Figures: currently produced through `studies/online_comparison`

Run from the repository root:

```bash
python studies/online_bnn/scripts/run_bnn_online_experiment.py --backend server
```

For online continuous runs, BNN acquisition does not reuse a fixed candidate
pool. Before each BNN-guided selection it resamples a fresh bounded batch,
snaps geometries to the configured resolution, predicts responses for that
batch, and selects the best EI-style candidate.

Use `--force-retrain-bnn` only when intentionally rebuilding BNN checkpoints.
