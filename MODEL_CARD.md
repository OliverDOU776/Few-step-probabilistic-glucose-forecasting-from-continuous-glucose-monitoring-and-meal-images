# Model card: GlucoFlow Stage-A initialization

## Artifact

- Path: `checkpoints/stage_a.pt`
- Format: PyTorch state dictionary (`weights_only=True` compatible)
- SHA-256: `759dabb7a5b20f9377a8d9ddaa2ae333d8e7166bc07c8d17a04f14d58acdd5a7`
- Size: approximately 2.6 MB

## Model

The checkpoint initializes the default `FlowMatchingModel`, a conditional rectified-flow forecaster with:

- a transformer temporal backbone;
- 24 historical time points and 24 forecast time points under the released defaults;
- optional time features and conditioning embeddings;
- Euler sampling with NFE=1, 2, or 4.

Stage A uses the learned null conditioning token. The checkpoint is therefore a temporal initialization rather than a final photo-conditioned or target-calibrated model.

## Training sources

The associated study describes Stage-A training on the training portions of:

- Weinstock through the GlucoBench preprocessing path;
- BIG IDEAs Lab;
- HUPA-UCM.

The paper reports 241 participants and approximately 795,000 CGM readings across those sources. Raw records are not embedded intentionally, but learned weights may still retain statistical information about their training distributions.

## Intended use

- Initialization for non-clinical time-series forecasting research.
- Cross-dataset pretraining followed by target-domain adaptation.
- Method-development studies of few-step probabilistic generation.
- Transfer to a new dataset after replacing the adapter, normalization, windowing, and evaluation logic.

## Required adaptation steps

Before use on a new target dataset:

1. define a participant- or entity-disjoint evaluation protocol where appropriate;
2. fit normalization using training data only;
3. adapt or fine-tune the model on the target domain;
4. select checkpoints and uncertainty parameters using validation data only;
5. evaluate once on a held-out test split;
6. establish application-specific safety and reliability requirements.

The checkpoint should not be treated as a calibrated predictor before those steps.

## Out-of-scope use

- Diagnosis, treatment, insulin dosing, alarms, or patient-facing decisions.
- Direct deployment without target-domain validation.
- Claims of demographic generalization, fairness, safety, or regulatory validation.
- Use as a final multimodal model without Stage-B adaptation.
- Interpreting sampling speed as evidence of clinical reliability.

## Limitations

The source cohorts are limited in population and sensor coverage. Dataset shift, missingness, meal timing, medications, insulin, activity, illness, and unobserved variables can invalidate forecasts. The released checkpoint has not been audited for privacy leakage, membership inference, adversarial robustness, or regulated-device use.
