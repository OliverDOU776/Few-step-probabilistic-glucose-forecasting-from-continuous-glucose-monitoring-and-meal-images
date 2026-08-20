# Third-party software, models, and data

No raw health dataset, meal photograph collection, downloaded model weight, or controlled-access record is distributed in this repository. Users are responsible for obtaining all external resources from their owners and complying with the applicable licenses, data-use agreements, and access conditions.

## OpenCLIP and LAION weights

The optional image-conditioning path uses [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) and the public [LAION ViT-B/32 model](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K). Their software, model, and training-data terms remain separate from this repository. Downloaded weights and derived image embeddings must not be committed here.

## Datasets in the associated study

The associated paper uses data obtained separately from their official sources:

- [CGMacros 1.0.0](https://physionet.org/content/cgmacros/1.0.0/) for multimodal adaptation and evaluation;
- [GlucoBench](https://github.com/IrinaStatsLab/GlucoBench) and its underlying cohorts for cross-cohort forecasting;
- [BIG IDEAs Lab](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/) and [HUPA-UCM](https://data.mendeley.com/datasets/3hbcscwz44/1) for temporal pretraining;
- [OhioT1DM](https://webpages.charlotte.edu/rbunescu/ohiot1dm.html) for controlled clinical-event evaluation;
- the **official controlled-access DiaTrend cohort** distributed through [Synapse](https://www.synapse.org/Synapse:syn38187184) for retrospective event-prediction evaluation.

Each underlying dataset retains its own terms. In particular, OhioT1DM and DiaTrend require controlled access and must not be redistributed from this code repository. The associated study used official DiaTrend data; no DiaTrend records or derived subject-level artifacts are included here.

## GlucoBench formatter

The reference pretraining code can load a user-provided checkout of [IrinaStatsLab/GlucoBench](https://github.com/IrinaStatsLab/GlucoBench). It is not vendored in this repository, and the licenses of the underlying datasets must be checked individually.

## README architecture figure

`assets/overview.png` is retained only to explain the software architecture. It is derived from the associated paper source. Any meal-image material in that figure originates from CGMacros and remains subject to the dataset's attribution and non-commercial share-alike terms.

## License separation

Dataset licenses, model licenses, and controlled-access approvals do not transfer to the GlucoFlow source code. Before releasing a derived model or cached embedding, verify the terms of every source dataset and pretrained model used to create it.
