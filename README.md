# Master Thesis Code

This repository contains the code used for the experiments, analysis, and result generation of my master thesis on transformation-aware visual representation learning using the MAPS dataset.

The repository is organized by the main stages of the project: dataset generation, model training, performance evaluation, and representation analysis.

---

## Repository Structure

```text
.
├── MAPS_Training/
├── MAPS_generation/
├── Representation Analysis/
├── Training and Testing Performance/
└── README.md
```

---

## Folder Overview

### `MAPS_generation/`

This folder contains the code related to the generation and preparation of the MAPS dataset.

Use this folder if you are looking for files related to:

- MAPS image generation
- Dataset construction
- Object categories and instances
- Transformation parameters
- MAPS indices and metadata
- Preprocessing steps for generated data

This part of the repository is mainly relevant for understanding how the synthetic MAPS data used in the experiments was created and organized.

---

### `MAPS_Training/`

This folder contains the code used for training the different model types evaluated in the thesis.

Use this folder if you are looking for files related to:

- Supervised model training
- Self-supervised learning training
- Action-based training
- ResNet18 and ResNet50 experiments
- Training configurations
- Model checkpoints or saved training outputs
- Training logs and metrics

This folder is the starting point for reproducing the trained models used in the later evaluation and analysis steps.

---

### `Training and Testing Performance/`

This folder contains scripts and results related to model performance evaluation.

Use this folder if you are looking for files related to:

- Training accuracy and loss plots
- Test accuracy plots
- Train/test comparison plots
- Accuracy across object instances
- Accuracy across transformations
- ImageNet transfer experiments
- Performance comparisons between supervised, self-supervised, and action-based models
- Action parameter ablation analysis

Most plots in the thesis that evaluate classification performance, generalization performance, or robustness are generated from this folder.

Example result types:

- Mean train/test accuracy across epochs
- Accuracy distributions over object instances
- Accuracy differences between training and test instances
- Robustness across individual transformation factors
- ImageNet performance for different MAPS/ImageNet mixtures

---

### `Representation Analysis/`

This folder contains the code used to analyze the learned embedding spaces of the trained models.

Use this folder if you are looking for files related to:

- PCA analysis
- PaCMAP visualizations
- Embedding trajectories
- Cosine similarity between neighboring embeddings
- Transformation consistency in latent space
- Correlation between transformation parameters and embedding directions
- Action parameter ablation analysis
- Layer-wise representation analysis

This folder is mainly used for the representation-level results in the thesis. These analyses go beyond classification accuracy and investigate how different learning objectives organize visual transformations in the latent space.

Example result types:

- PCA plots over object transformations
- PaCMAP visualizations of embedding trajectories
- Adjacent cosine similarity plots
- Embedding principal component correlation analysis
- Comparison of supervised, self-supervised, and action-based representations

---

## Thesis Context

The experiments compare different learning objectives for visual representation learning on the MAPS dataset:

- Supervised learning
- Classical self-supervised learning
- Action-based / transformation-aware learning

The main goal is to investigate how these learning paradigms differ in terms of classification performance, generalization, robustness across visual transformations, and the geometric structure of the learned representations.

---

## Notes

This repository contains the code used for the final thesis results and plotting. Some scripts assume that trained models, extracted embeddings, or saved metric files already exist in the expected directory structure.

For reproducibility, the recommended order is:

1. Prepare the dataset.
2. Train the required models.
3. Run the performance evaluation scripts.
4. Run the representation analysis and visualization scripts.

Large generated files, intermediate outputs, model checkpoints, or datasets are not included in this repository but can be provided if requested.
