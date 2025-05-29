
# Implementations for Recommendation Systems

  

A PyTorch implementation of the SVDppAG model introduced in the paper "SVDppAG: Enhancing SVD++ with Attention and Gating".

  

## Overview

  

This project contains an implementation of the SVDppAG model introduced in the paper "SVDppAG: Enhancing SVD++ with Attention and Gating" which incoorporates explicit ratings and implicit feedback for user-item ratings prediction. Additionally, it includes implementations of state-of-the-art models for recommendation systems based on earlier papers for performance comparison.

  

## Features

  

### Models Implemented

  

- **SVD++ with Attention and Gating (SVDppAG)**: Advanced variant with attention mechanisms and gating networks

- **Basic Matrix Factorization (BasicMF)**: Standard collaborative filtering with user and item embeddings

- **SVD++**: Enhanced matrix factorization incorporating implicit feedback

- **Asymmetric SVD (AsymmetricSVD)**: Alternative approach to implicit feedback integration

- **Basic Implicit MF**: Matrix factorization specifically designed for implicit feedback

- **Alternating Least Squares (ALS)**: Traditional optimization approach based on matrix factorization

  

### Key Features of SVDppAG

  

- **Implicit Feedback Support**: All neural models support incorporation of implicit feedback signals

- **Flexible Architecture**: Modular design allowing easy experimentation with different model variants

- **GPU Acceleration**: Full CUDA support for training and inference

- **Batch Processing**: Efficient batch processing for both training and inference

  

## Installation

  

```bash

pip install torch pandas numpy scikit-learn tqdm

```

  

## Quick Start

  

### Running the Example

  

```bash

python SVDppMLP_sample_training.py

```

  

## Model Details

  

### SVD++ (Singular Value Decomposition Plus Plus)

  

The core SVD++ model predicts ratings using:

  

$\hat{r}_{ui} = μ + b_u + b_i + q_i^T(p_u + |N(u)|^(-1/2) * Σ_{j∈N(u)} y_j)$

  

Where:

- $μ$: Global mean rating

- $b_u$, $b_i$: User and item biases

- $p_u$, $q_i$: User and item latent factors

- $N(u)$: Set of items with implicit feedback from user u

- $y_j$: Implicit item factors

  

## Data Format

  

The system expects the following data formats where "sid" are item IDs and "pid" are user IDs:

  

### Explicit Ratings (`train_ratings.csv`)

```csv

sid_pid,rating

1_100,4.5

1_101,3.0

...

```

  

### Implicit Feedback (`train_tbr.csv`)

```csv

sid,pid

1,200

1,201

...

```

  

## Hyperparameter Tuning

  

Key hyperparameters to tune:

  

- `embedding_dim`: Dimensionality of latent factors (typically 20-200)

- `learning_rate`: Learning rate for optimization (typically 0.001-0.01)

- `reg_lambda`: L2 regularization strength (typically 0.001-0.1)

  

## Advanced Features

  

### Flexible Implicit Feedback

- Variable-length implicit feedback lists

- Automatic padding and masking

- Normalization by feedback count