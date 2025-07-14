
# LLM-Based Detection of Tangled Code Changes for Higher-Quality Method-Level Bug Datasets

## 🧠 Overview

This project formulates **method-level tangled change detection** as a binary classification task: determine whether a method-level code diff is **Buggy** (related to a bug fix) or **NotBuggy**.

The approach leverages:
- Prompt-based reasoning with GPT-4o, GPT-4o-mini, and Gemini 2.0 Flash
- Embedding-based classifications with models like Multi-Layer Perceptron (MLP)


## 📁 Dataset

### Gold Standard Dataset

- Location: [Complete_GoldSet.csv](./data/Complete_GoldSet.csv)
- Total Samples: 1,764 method-level code diffs
- Labels: `Buggy`, `NotBuggy`
- Composition:
  - **Phase 1**: 1,457 untangled samples
  - **Phase 2**: 307 manually annotated tangled examples

Each sample includes:
- Method-level code diff
- Commit message

### Bug Dataset
- [49 Java open-source projects](https://github.com/shaifulcse/dataset-MLBP-2022) dataset by Chowdhury et al.
- Noisy Method-level bug dataset at method's inception: [NoisyDataset](./data/NoisyDataset/)
- Noise-Fre Method-level bug dataset at method's inception: [NoiseFreeDataset](./data/NoiseFreeDataset//)


## 🔬 Reproducing Paper Results

### RQ1
- Run [RQ1.ipynb](./RQ1.ipynb) to reproduce the results of RQ1.
- Results are stored at [./Results/RQ1/](./Results/RQ1/)

### RQ2
- Run [RQ2.ipynb](./RQ2.ipynb) to reproduce the results of RQ2.
- Results are stored at [./Results/RQ2/](./Results/RQ2/)

### RQ3
- Run [RQ3.ipynb](./RQ3.ipynb) to reproduce the results of RQ3.
- Results are stored at [./Results/RQ3/](./Results/RQ3/)

### RQ4
- Run [RQ4.ipynb](./RQ4.ipynb) to reproduce the results of RQ4.
- Results are stored at [./Results/RQ4/](./Results/RQ4/)
