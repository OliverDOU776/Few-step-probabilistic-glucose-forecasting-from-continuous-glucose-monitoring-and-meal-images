# Reported-result snapshot

These CSVs are a compact snapshot selected from the manuscript-era research workspace. They contain no subject-level rows, raw CGM, meal photographs, embeddings, checkpoints, or per-window predictions.

## Files

- `main/cgmacros_p1.csv`: CGMacros P1 metrics for the promoted anchor-assisted configuration.
- `main/glucobench_accuracy.csv`: GlucoBench accuracy table.
- `main/glucobench_uncertainty.csv`: GlucoBench uncertainty table.
- `main/clinical_events.csv`: the provenance-verified OhioT1DM portion of the clinical table.
- `main/cgmacros_auc_reference.csv`: an additional metric-family comparison with protocol notes.
- `ablations/pretraining_and_photos.csv`: subject-disjoint pretraining/photo ablation.
- `provenance/glucobench_selected_runs.csv`: configurations associated with the archived GlucoBench cells.

## Caveats

1. The original GlucoBench aggregate was selected from a broad research sweep and did not retain a complete fixed three-seed manifest. The public aggregate implementation now selects configurations using validation metrics only.
2. The CGMacros P1 point forecast is ridge-anchor assisted; the rectified flow supplies stochastic residual trajectories. The table must not attribute the anchor's point accuracy to the flow alone.
3. The CSVs do not contain standard-deviation columns and must not be described as independently reconstructing manuscript mean +/- standard deviation values.
4. A DiaData integration was previously mislabeled as controlled-access DiaTrend. That unsupported row is omitted. See `docs/DATA.md`.

Treat this directory as transparent report provenance, not as raw experimental evidence or a replacement for rerunning the public code on properly licensed data.
