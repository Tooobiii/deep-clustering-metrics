# Deep Clustering Metrics

This repository contains the code developed for my Bachelor's Thesis
"*Evaluating Internal Clustering Metrics for Effective Architecture Recommendation in Deep Clustering*".

The project investigates how well different internal clustering validity indices (CVIs) evaluate the quality of
deep clustering results. It compares the scores produced by these indices with external clustering metrics and uses
correlation analyses to identify how closely they agree.

## Requirements

Install the project and its dependencies with:

`python -m pip install .`

## Configuration and Customization

The experiments can be customized through command-line arguments and configuration files.
You can check the available arguments in the individual scripts and `args.py`.

**Ensure that input files and output directories match the expected project structure**.

## Pipeline

All programs are expected to be run from the root project folder.

1. (Optional) **Create a subset of the Reuters dataset.**

`python utils/prep_reuter.py --subset_size <N>`

In the experiments, subsets of the Reuters dataset were used due to its large size.

2. **Benchmark one run**

`python main.py --model <"MODEL"> --dataset <"DATASET">`

Repeat this step with different configurations.
3. **Preprocess results**

`python results/preprocess_csv.py`

Cleans and transforms the results, optionally aggregating repeated configurations.

4. **Compute correlations**

`python results/correlation.py --target <"TARGET">`

Computes Spearman and Kendall correlations between the selected external metric and CVIs, both globally and for
selected subsets.

## Further analysis

The generated result tables serve as the basis for the statistical analyses presented in the thesis.
The embeddings produced during the experiments were used separately to generate visualizations and plots.

## Project Status

The project is considered complete. Minor improvements, such as adding further docstrings and more detailed help
descriptions for command-line arguments, may be made in the future