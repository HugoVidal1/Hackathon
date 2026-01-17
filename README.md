# Voice-Based Heart Failure Prediction Challenge

## Overview

In this challenge, you will develop **patient-specific, temporal-aware representations (embeddings)** of voice features to identify recordings acquired within a 4-week window preceding a heart failure (HF)-related hospitalization.

## The Task

Each patient in the dataset has been followed longitudinally over several months with regular voice recordings. Your goal is to:

1. **Learn meaningful embeddings** from pre-computed acoustic feature vectors
2. **Detect progressive voice degradation** that signals impending hospitalization
3. **Use only a simple linear classifier** on your embeddings (forcing you to create rich representations)

Each recording is labeled as:
- **1**: Recording falls within the 4-week pre-hospitalization window
- **0**: Recording is from a stable period

## Dataset Structure

The dataset (`data/dataset.parquet`) contains:
- **73,542 recordings** from **14 patients**
- **~780 acoustic features** per recording (MFCCs, spectral features, jitter, shimmer, etc.)
- **Metadata**: `patient_short_id`, `recording_id`, `start_time`, `end_time`, `age`, `sex`, `audio_quality`
- **Target**: `label` (binary: 0 or 1)

## Evaluation

The evaluation uses **Leave-One-Patient-Out (LOPO)** cross-validation to ensure:
- No data leakage between train and test sets
- Patient-specific patterns are learned, not patient identity

**Metric**: Mean and standard deviation of per-patient ROC AUC scores.

## Repository Structure

```
challenge_repo/
├── data/
│   └── dataset.parquet          # Your dataset
├── src/
│   ├── models.py                # Define your embedding model here
│   ├── train.py                 # Training loop with LOPO enforcement
│   └── utils.py                 # Evaluation metrics and visualization tools
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Understand the Code Structure

- **`src/models.py`**: Contains a simple `EmbeddingModel` template. Implement your embedding architecture here.
- **`src/train.py`**: Implements the LOPO training loop with a forced linear head. Run this to train and evaluate.
- **`src/utils.py`**: Utility functions for computing metrics and visualizing embeddings.
- **`src/config.py`**: Pydantic-based configuration with automatic validation. Modify hyperparameters here.

### 3. Run the Baseline

```bash
python src/train.py
```

This will train a simple baseline model using LOPO cross-validation and display:
- Per-patient ROC AUC scores
- Mean and standard deviation of ROC AUC across all patients
- 2D PCA visualization of learned embeddings

### 4. Improve the Model

**First, tune hyperparameters** in `src/config.py`:
- Change `embedding_dim`, `hidden_dims`, `batch_size`, `num_epochs`, `learning_rate`, etc.
- Pydantic will validate your values automatically!

**Then, focus on architecture** in `src/models.py`:
- **Temporal modeling**: Voice degradation happens over time. Can you capture this?
- **Patient-specific features**: Each patient has unique baseline voice characteristics.
- **Embedding quality**: The linear head is fixed, so your embeddings must be discriminative.

Some ideas:
- Use recurrent architectures (LSTM, GRU) to model temporal sequences
- Implement attention mechanisms to weight important time points
- Add patient-specific normalization or adaptation layers
- Experiment with contrastive learning or metric learning objectives

## Important Notes

### LOPO Enforcement
The training loop ensures no patient's data appears in both train and test sets. **Do not modify this behavior** as it would invalidate your results.

### Linear Head Constraint
The classifier head is a simple linear layer. This constraint ensures you focus on learning rich embeddings rather than relying on a complex classifier.

### Temporal Awareness
Recordings have `start_time` and `end_time` information. Use this to model the temporal progression of voice changes!

## Tips

1. **Start simple**: Get the baseline running first, then iterate.
2. **Visualize embeddings**: Use the PCA plot to see if your embeddings separate positive/negative samples.
3. **Monitor per-patient performance**: Some patients may be harder to predict than others.
4. **Consider data imbalance**: Check the class distribution in your training data.

## Good Luck!

Remember: The goal is to create embeddings that capture the subtle, progressive changes in voice that precede hospitalization. Think about what makes voice change over time and how you can model that effectively.

