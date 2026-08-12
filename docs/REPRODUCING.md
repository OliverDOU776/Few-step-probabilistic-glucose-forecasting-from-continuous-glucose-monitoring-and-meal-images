# Reproducing GlucoFlow experiments

## Scope

| Track | Public entry point | Current boundary |
|---|---|---|
| Core rectified-flow model | `scripts/smoke_test.py`, unit tests | Data-free and deterministic on CPU |
| Stage-A pretraining | `scripts/pretrain.py` | Requires Weinstock, BIG IDEAs, and HUPA-UCM |
| CGMacros data + CLIP cache | `scripts/prepare_data.py` | Requires CGMacros and OpenCLIP weights |
| Subject-disjoint CGMacros | `scripts/finetune_cgmacros.py` | Single seeded run; photo runs require cache |
| CGMacros P1 | `scripts/evaluate_cgmacros.py` | 10-fold run; point forecast is ridge-anchor assisted |
| GlucoBench | `scripts/evaluate_glucobench.py` | Fixed single runs; aggregate selection uses validation only |
| OhioT1DM | `scripts/evaluate_clinical.py` | Requires the official DUA-protected XML release |
| DiaTrend | — | Withheld pending controlled-data adapter and provenance-correct rerun |

## Environment

The paper-era environment used Python 3.10, PyTorch 2.8, CUDA 12.6, NumPy 1.26, pandas 2.3, and scikit-learn 1.7. The package metadata permits compatible versions rather than reproducing the unrelated 508-package workstation freeze.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[vision,dev]"
pytest -q
```

GPU training is recommended. CPU execution is supported for unit tests and the checkpoint smoke test.

## Stage A: CGM-only pretraining

Prepare the three datasets according to [DATA.md](DATA.md), then run:

```bash
python scripts/pretrain.py \
  --seed 42 --split-seed 42 \
  --epochs 20 --batch-size 256 \
  --max-train-windows-per-dataset 20000 \
  --max-val-windows-per-dataset 8000
```

The best validation checkpoint is written to `checkpoints/stage_a.pt`. This overwrites the distributed initialization checkpoint, so copy it first if you want to retain both.

## Stage B: multimodal CGMacros

Generate image embeddings and data cards:

```bash
python scripts/prepare_data.py
```

Run one subject-disjoint configuration:

```bash
python scripts/finetune_cgmacros.py \
  --mode run --config-name release_p2 \
  --seed 42 --split-seed 42 \
  --epochs 36 --warmup-epochs 4 \
  --image-drop-prob 0.15 --nutrient-drop-prob 0.05
```

Run the P1 direct task:

```bash
python scripts/evaluate_cgmacros.py \
  --mode run --config-name p1_anchor_photo \
  --use-photo --seed 42 --epochs 20 --patience 5 \
  --n-folds 10 --alignment-weight 0.0

python scripts/evaluate_cgmacros.py --mode aggregate
```

The P1 runner uses a deterministic ridge point anchor and a conditional flow for residual trajectories. Consequently, its point metrics must be described as anchor-assisted; the probabilistic samples come from the flow model.

## GlucoBench

After pinning and extracting GlucoBench as described in [DATA.md](DATA.md):

```bash
for dataset in iglu colas dubosson hall weinstock; do
  python scripts/evaluate_glucobench.py \
    --mode run --dataset "$dataset" \
    --config-name "release_${dataset}" \
    --seed 42 --split-seed 10 \
    --use-anchor --warm-start --in-len-source linreg
done

python scripts/evaluate_glucobench.py --mode aggregate
```

Within a run, point candidates and uncertainty scale are selected on validation data. Across runs, the public aggregate path also ranks configurations using stored validation metrics before it reports held-out test metrics.

## OhioT1DM

```bash
python scripts/evaluate_clinical.py \
  --mode run --dataset ohio_t1dm \
  --config-name release_ohio --seed 42

python scripts/evaluate_clinical.py --mode aggregate
```

Ohio uses the official `ws-training` / `ws-testing` boundary, with the final 20% of each official training series reserved for validation.

## Outputs

New runs write under `artifacts/`, which is ignored by Git:

- `artifacts/metrics/`: run payloads and logs
- `artifacts/tables/`: generated aggregate CSVs
- `artifacts/figures/`: generated plots
- `artifacts/run_summaries/`: human-readable summaries
- `artifacts/cache/`: derived CLIP embeddings

Only small, promoted manuscript-era snapshots live under `results/`.

## Important interpretation boundary

The original research workspace did not retain a complete fixed three-seed command manifest and seed-level output for every manuscript cell. Therefore:

- commands above reproduce the implemented protocols as new single runs;
- they do not promise byte-identical regeneration of the manuscript's mean +/- standard deviation values;
- the archived CSVs must not be presented as a preregistered fixed-config benchmark;
- latency is hardware- and implementation-specific and excludes CLIP preprocessing in the manuscript figure.
