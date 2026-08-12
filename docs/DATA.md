# Data access and layout

This repository never distributes raw CGM, meal, image, demographic, XML, CSV, Excel, or downloaded archive files. Download data only from the official sources below and follow their access and license terms.

| Dataset | Role | Official source | Access / license | Expected local root |
|---|---|---|---|---|
| CGMacros 1.0.0 | Multimodal fine-tuning and evaluation | [PhysioNet](https://physionet.org/content/cgmacros/1.0.0/) / [DOI](https://doi.org/10.13026/3z8q-x658) | Open access; CC BY-NC-SA 4.0 | `data/raw/cgmacros/` |
| GlucoBench | Five CGM benchmarks and official formatter | [GitHub](https://github.com/IrinaStatsLab/GlucoBench) | Public; underlying datasets retain separate terms | `external/GlucoBench/` |
| BIG IDEAs Lab 1.1.3 | Stage-A CGM-only pretraining | [PhysioNet](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/) / [DOI](https://doi.org/10.13026/aw6y-fc44) | Open access; ODC Attribution 1.0 | `data/raw/big_ideas/` |
| HUPA-UCM | Stage-A CGM-only pretraining | [Mendeley Data](https://data.mendeley.com/datasets/3hbcscwz44/1) / [DOI](https://doi.org/10.17632/3hbcscwz44.1) | Public download; CC BY 4.0 | `data/raw/hupa_ucm/` |
| OhioT1DM | Controlled clinical-event evaluation | [Dataset page](https://webpages.charlotte.edu/rbunescu/ohiot1dm.html) / [DUA form](https://ohio.qualtrics.com/jfe/form/SV_02QtWEVm7ARIKIl) | Signed DUA required; research use; no redistribution | `data/raw/ohio_t1dm/` |
| DiaTrend | Not currently supported | [Synapse](https://www.synapse.org/Synapse:syn38187184) / [DOI](https://doi.org/10.7303/syn38187184) | Controlled access; Synapse certification and intended-use approval | `data/raw/diatrend_official/` |

## Expected structures

### CGMacros

The adapter expects the dateshifted release at:

```text
data/raw/cgmacros/
└── cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0/
    └── CGMacros_dateshifted365/
        └── CGMacros/
            ├── bio.csv
            ├── gut_health_test.csv
            ├── CGMacros-001/
            └── ...
```

After extraction, generate frozen OpenCLIP ViT-B/32 features with:

```bash
python scripts/prepare_data.py
```

The cache is written to `artifacts/cache/clip_embeddings/cgmacros/` and is ignored by Git. Cache filenames contain subject and meal timestamps, so they must not be published even though the underlying dataset is public.

### GlucoBench

Clone the upstream implementation and pin the version used in this project:

```bash
git clone https://github.com/IrinaStatsLab/GlucoBench.git external/GlucoBench
git -C external/GlucoBench checkout 661d840a98b316df51faa13a7100430afcbbb5b7
unzip external/GlucoBench/raw_data.zip -d external/GlucoBench
```

The runners expect YAML files in `external/GlucoBench/config/`, formatter code in `external/GlucoBench/data_formatter/`, and extracted CSVs in `external/GlucoBench/raw_data/`.

GlucoBench does not provide a single top-level license that can be assumed for every dataset. Follow each underlying dataset's terms and do not redistribute its archive from this repository.

### BIG IDEAs Lab

Keep the official subject directories under `data/raw/big_ideas/`. The adapter searches recursively for `Dexcom_*.csv`, so no flattening is required. Stage A uses CGM only.

### HUPA-UCM

Place the extracted `HUPA####P.csv` files anywhere below `data/raw/hupa_ucm/`. The adapter searches recursively and uses the CGM and available time-varying covariates.

### OhioT1DM

After completing the official DUA process, place the six-subject XML release directly under:

```text
data/raw/ohio_t1dm/
├── 540-ws-training.xml
├── 540-ws-testing.xml
└── ...
```

The parser uses the release's `dd-mm-yyyy HH:MM:SS` timestamp convention and preserves the official per-subject training/testing boundary.

## DiaTrend and DiaData are not interchangeable

Official **DiaTrend** contains data from 54 people with type 1 diabetes and is distributed as controlled-access workbooks through Synapse. The current public repository does not contain a verified adapter or result for it.

The separate [DiaData project](https://github.com/Beyza-Cinar/DiaData) ([Zenodo DOI](https://doi.org/10.5281/zenodo.16874128)) is a public integration of 13 source datasets covering 1,720 individuals under CC BY-NC 4.0. Its public release explicitly excludes restricted DiaTrend data. A prior internal workspace mislabeled DiaData CSVs as DiaTrend; this release removes that adapter and result rather than perpetuating the provenance error.

## OpenCLIP weights

Image embeddings use `open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")` from [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip). The corresponding model card is [laion/CLIP-ViT-B-32-laion2B-s34B-b79K](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K). No Hugging Face token is required for this public model. Keep downloaded weights in the default model cache; never commit them here.
