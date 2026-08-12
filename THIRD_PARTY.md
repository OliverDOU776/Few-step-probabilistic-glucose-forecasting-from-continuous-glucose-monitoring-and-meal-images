# Third-party software, models, data, and figures

No third-party dataset or raw meal photograph is distributed as a standalone file in this repository. Users are responsible for obtaining data and complying with the source terms in `docs/DATA.md`.

## OpenCLIP

The optional image pipeline uses [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) and the public [LAION ViT-B/32 model](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K). Consult the software repository and model card for their licenses and training-data limitations. Downloaded weights and derived embeddings are excluded from Git.

## GlucoBench

The benchmark formatter is loaded from a user-provided checkout of [IrinaStatsLab/GlucoBench](https://github.com/IrinaStatsLab/GlucoBench), pinned in the documentation to commit `661d840a98b316df51faa13a7100430afcbbb5b7`. It is not vendored here. Each underlying dataset retains its own terms.

## README figures and CGMacros material

The README figures are copied from the associated [paper source repository](https://github.com/OliverDOU776/paper-Few-Step). The overview figure contains meal-image material derived from [CGMacros](https://physionet.org/content/cgmacros/1.0.0/), released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Source dataset citation: Thomaz et al., *CGMacros: a scientific dataset for personalized nutrition and diet monitoring*, DOI [10.13026/3z8q-x658](https://doi.org/10.13026/3z8q-x658).

## Dataset terms

Dataset licenses and access agreements do not transfer to this code repository. In particular, OhioT1DM and official DiaTrend are controlled-access and may not be redistributed here.
