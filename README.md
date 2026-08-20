<div align="center">

# GlucoFlow

### Reusable few-step probabilistic forecasting with cross-dataset pretraining and optional multimodal conditioning

[![Paper](https://img.shields.io/badge/associated%20paper-source-4C78A8.svg)](https://github.com/OliverDOU776/new-paper-fewstep)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-3776AB.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C.svg)
![Status](https://img.shields.io/badge/status-research%20software-2A9D8F.svg)

Conditional rectified flow, transferable temporal pretraining, and optional image--nutrient conditioning.

<img src="assets/overview.png" width="100%" alt="Overview of GlucoFlow" />

</div>

> [!IMPORTANT]
> GlucoFlow is research software, not a medical device. It must not be used for diagnosis, insulin dosing, treatment, alarms, or any other clinical decision.

## Purpose of this repository

This repository provides the reusable implementation of GlucoFlow. It is organized for researchers and engineers who want to:

- initialize a forecasting model from a CGM-only temporal checkpoint;
- pretrain on one or more time-series datasets before adapting to a smaller target dataset;
- generate probabilistic future trajectories with 1, 2, or 4 Euler steps;
- condition forecasts on optional side information such as meal photographs or nutrient vectors;
- replace the supplied dataset adapters with an adapter for another dataset or application.

Paper-specific result tables, sweep summaries, run-selection records, and manuscript figures are intentionally not stored here. The associated paper source and full experimental reporting are maintained separately in [new-paper-fewstep](https://github.com/OliverDOU776/new-paper-fewstep).

## Core design

| Component | Role |
|---|---|
| **Conditional rectified flow** | Generates a distribution of future trajectories with a small number of Euler steps. |
| **Cross-dataset temporal pretraining** | Learns transferable dynamics from larger time-series cohorts before target-domain adaptation. |
| **Optional conditioning** | Accepts a learned conditioning vector, including image, nutrient, metadata, or null representations. |
| **Missing-modality fallback** | Supports combined, single-modality, or null conditioning paths. |

The core model consumes normalized history tensors and optional conditioning embeddings. Dataset parsing, normalization, window extraction, and application-specific metrics are deliberately separated from the flow model so that they can be replaced for a new use case.

## Installation

```bash
git clone https://github.com/OliverDOU776/Few-step-probabilistic-glucose-forecasting-from-continuous-glucose-monitoring-and-meal-images.git
cd Few-step-probabilistic-glucose-forecasting-from-continuous-glucose-monitoring-and-meal-images

python3.11 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[vision]"
```

For development:

```bash
python -m pip install -e ".[vision,dev]"
pytest -q
```

## Verify the released checkpoint

The repository includes the Stage-A CGM-only initialization checkpoint at `checkpoints/stage_a.pt`. Run the data-free smoke test:

```bash
python scripts/smoke_test.py
```

The test loads the checkpoint, samples trajectories with NFE=1, 2, and 4, and checks output shapes and finite values.

## Minimal programmatic inference

Inputs must be normalized with statistics fitted on the training split of the target application. The example below demonstrates the tensor interface; it does not perform clinical preprocessing.

```python
from pathlib import Path

import torch

from glucoflow.models import build_flow_model

model = build_flow_model(history_len=24, prediction_len=24)
state = torch.load(
    Path("checkpoints/stage_a.pt"),
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(state, strict=True)
model.eval()

# Two normalized histories, 24 time points each.
history = torch.randn(2, 24)
time_features = torch.zeros(2, 24, 4)

with torch.no_grad():
    samples = model.sample(
        history,
        nfe=4,
        num_samples=100,
        time_features=time_features,
        meal_embed=None,
    )

# [batch, sampled trajectories, forecast length]
assert samples.shape == (2, 100, 24)
median = samples.median(dim=1).values
lower = samples.quantile(0.05, dim=1)
upper = samples.quantile(0.95, dim=1)
```

For a multimodal application, pass a target-domain conditioning vector through `meal_embed`. The supplied `MealEncoder` shows the image--nutrient implementation used in the associated study, but the flow model only requires a tensor of the configured conditioning dimension.

## Adapt GlucoFlow to a new dataset

### 1. Implement the shared data contract

`glucoflow.data.adapters.base.DatasetOutput` defines the adapter output. At minimum, the CGM/time-series table contains:

```text
subject_id | timestamp | glucose_mgdl
```

The names reflect the original application; for another domain, map the target series into `glucose_mgdl` or fork the schema with a domain-specific name. Optional meal/conditioning and subject tables may be returned alongside the time series.

A complete CSV example is provided in [`examples/custom_adapter.py`](examples/custom_adapter.py):

```bash
python examples/custom_adapter.py /path/to/my_dataset
```

The example expects `/path/to/my_dataset/cgm.csv`, validates the required columns, parses timestamps, sorts each subject chronologically, and estimates the sampling interval.

### 2. Choose a training path

- Use `scripts/pretrain.py` as a reference for multi-dataset Stage-A pretraining.
- Use `scripts/finetune_cgmacros.py` as a reference for target-domain adaptation with optional conditioning and modality dropout.
- Use the reusable modules in `src/glucoflow/` directly when building a cleaner application-specific trainer.

The scripts retain the original glucose-study defaults, so a new application should explicitly replace dataset adapters, clipping rules, normalization, history/forecast lengths, and evaluation metrics.

### 3. Keep validation and test roles separate

Fit normalization, select checkpoints, choose uncertainty scales, and tune operating parameters on training/validation data only. Evaluate the held-out test split once after the configuration is fixed.

## Repository layout

```text
.
├── src/glucoflow/
│   ├── models/              # Rectified-flow backbone, sampling, meal encoder
│   ├── data/                # Shared schema, normalization, adapters
│   └── evaluation/          # Splits, windows, and reusable metrics
├── scripts/
│   ├── prepare_data.py      # Reference data preparation and feature caching
│   ├── pretrain.py          # Reference Stage-A training entry point
│   ├── finetune_cgmacros.py # Reference Stage-B multimodal adaptation
│   ├── smoke_test.py        # Data-free checkpoint test
│   └── validate_release.py  # Release-integrity checks
├── examples/
│   └── custom_adapter.py    # Template for a new dataset
├── checkpoints/stage_a.pt   # Released temporal initialization
├── tests/                   # Data-free unit and integrity tests
└── assets/overview.png      # Architecture overview
```

Raw health data, meal photographs, derived embeddings, run logs, experiment sweeps, and paper result snapshots are not distributed.

## Datasets used in the associated study

The associated paper uses:

- CGMacros for multimodal adaptation and participant-disjoint evaluation;
- Weinstock, BIG IDEAs, and HUPA-UCM for CGM-only pretraining;
- Broll, Colas, Dubosson, Hall, and held-out Weinstock participants for cross-cohort forecasting evaluation;
- OhioT1DM and the **official controlled-access DiaTrend cohort** for retrospective event-prediction evaluation.

This repository does not redistribute any of those datasets. Official DiaTrend is accessed through Synapse under its own approval and use conditions; the controlled files and paper-specific ingestion pipeline are not included in the public release. Users adapting GlucoFlow to DiaTrend or another controlled dataset should implement a local adapter without committing source records.

## Checkpoint and limitations

`checkpoints/stage_a.pt` contains the CGM-only Stage-A state dictionary.

```text
SHA-256  759dabb7a5b20f9377a8d9ddaa2ae333d8e7166bc07c8d17a04f14d58acdd5a7
```

The checkpoint is an initialization, not a calibrated model for a new population. Target-specific normalization, adaptation, validation, and safety assessment remain necessary. See [MODEL_CARD.md](MODEL_CARD.md) for intended and out-of-scope uses.

## Citation

```bibtex
@misc{wang2026glucoflow,
  title  = {GlucoFlow: Cross-Dataset Pretraining for Few-Step
            Probabilistic Glucose Forecasting},
  author = {Wang, Zijia and Toumazou, Christofer},
  year   = {2026},
  note   = {Research software and associated manuscript},
  url    = {https://github.com/OliverDOU776/Few-step-probabilistic-glucose-forecasting-from-continuous-glucose-monitoring-and-meal-images}
}
```

## Third-party components and data terms

The optional image path uses OpenCLIP and public LAION ViT-B/32 weights. Dataset licenses and controlled-access agreements do not transfer to this repository. See [THIRD_PARTY.md](THIRD_PARTY.md) before redistributing derived artifacts.
