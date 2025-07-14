
# LLM-Based Detection of Tangled Code Changes for Higher-Quality Method-Level Bug Datasets

Tangled code changes—commits that conflate unrelated modifications such as bug fixes, refactorings, and enhancements—introduce significant noise into bug datasets and adversely affect the performance of bug prediction models. Addressing this issue at a fine-grained, method-level granularity remains underexplored. This is critical to address, as recent bug prediction models, driven by practitioner demand, are increasingly focusing on finer granularity rather than traditional class- or file-level predictions. This study investigates the utility of Large Language Models (LLMs) for detecting tangled code changes by leveraging both commit messages and method-level code diffs. We formulate the problem as a binary classification task and evaluate multiple prompting strategies, including zero-shot, few-shot, and chain-of-thought prompting, using state-of-the-art proprietary LLMs such as GPT-4o and Gemini-2.0-Flash. 

Our results demonstrate that combining commit messages with code diffs significantly enhances model performance, with the combined few-shot and chain-of-thought prompting achieving an F1-score of 0.88. Additionally, we explore machine learning models trained on LLM-generated embeddings, where a multi-layer perceptron classifier achieves superior performance (F1-score: 0.906, MCC: 0.807). Applying our approach to 49 open-source projects improves the distributional separability of code metrics between buggy and non-buggy methods, enhancing dataset quality. These findings demonstrate the promise of LLMs for method-level commit untangling and contribute to improving the reliability of bug prediction models.

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
