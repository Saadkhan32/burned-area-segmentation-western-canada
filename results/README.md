# Results

- `runs/<configuration>_<fold>/config.json` - every argument of the run, the training, validation and test windows, the torch version; `norm_stats.json` - per-band normalisation from the training windows of the fold; `log.csv` - per-epoch training loss and validation F1; `ledger.json` - best validation F1, epochs run, SHA-256 of `best.pt`, wall time, GPU. The `_C2` runs are the plain-cross-entropy contrast.
- `runs/run_ledger.csv` - one line per run: run, best validation F1, epochs, checkpoint SHA-256, finish time, wall seconds, GPU.
- `per_window/<run>_test_windows.csv` - per test window: confusion counts over valid pixels, precision, recall, F1, IoU, mapped and reference burned area, for every configuration and for the label-free rules.
- `summary/` - `summary_by_model.csv` (mean per-window F1 with bootstrap intervals), `paired_differences.csv` (per-window paired differences, sign-flip tests, Holm), `all_windows.csv`, `negative_windows_by_model.csv`, `summary.md`.
- `tables/` - the tables of the paper and the supplement as written by `code/analysis/make_figures_planA.py` (CSV and Markdown).
- `inference_benchmark/` - `benchmark_inference_F-AB.csv` (seconds per window, tiles per second, GPU memory; five Alberta test windows, one A100) and the job output.
- `slurm_logs/` - standard output of the 42 training jobs.
