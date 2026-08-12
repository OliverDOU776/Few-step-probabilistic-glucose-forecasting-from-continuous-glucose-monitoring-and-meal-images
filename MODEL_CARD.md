# Model card: Stage-A GlucoFlow checkpoint

## Artifact

- Path: `checkpoints/stage_a.pt`
- Format: PyTorch state dictionary (`weights_only=True` compatible)
- SHA-256: `759dabb7a5b20f9377a8d9ddaa2ae333d8e7166bc07c8d17a04f14d58acdd5a7`
- Size: approximately 2.6 MB

## Model

The checkpoint initializes the default `FlowMatchingModel`: a conditional rectified-flow forecaster with a transformer backbone, 120 minutes of 5-minute CGM history (24 steps), a 120-minute output (24 steps), and 1/2/4-step Euler sampling. Stage A uses the learned null meal token; it is not itself a photo-conditioned model.

## Training data

The manuscript describes Stage-A training on subject-disjoint training portions of:

- Weinstock through the GlucoBench preprocessing path;
- BIG IDEAs Lab;
- HUPA-UCM.

The manuscript snapshot reports 241 training subjects and approximately 795,000 CGM readings. No raw records are embedded intentionally, but model weights can still retain statistical information about their training distribution.

## Intended use

- Initialization for the research training and evaluation scripts in this repository.
- Reproduction studies of few-step probabilistic time-series forecasting.
- Non-clinical methodological research.

## Out-of-scope use

- Diagnosis, treatment, insulin dosing, alarms, or patient-facing decisions.
- Use as a calibrated predictor without dataset-specific normalization and evaluation.
- Claims of demographic generalization, fairness, safety, or regulatory validation.
- Direct use as the final multimodal model; Stage-B adaptation is required for meal conditioning.

## Limitations

The source cohorts are limited in size and population coverage. Dataset shifts, sensor differences, missingness, meal timing, medications, insulin, and unobserved clinical variables can invalidate forecasts. Sampling speed does not imply clinical reliability. The checkpoint has not been audited for privacy leakage or membership inference.
