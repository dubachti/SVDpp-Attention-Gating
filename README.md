# SVDppAG: Enhancing SVD++ with Attention and Gating

This repository contains the implementation for the methods described in the [project report](docs/SVDppAG.pdf).

## Overview

**SVDppAG** extends the SVD++ collaborative filtering model by introducing attention and gating mechanisms to improve the aggregation and fusion of implicit feedback signals.

- **Attention Mechanism**: Dynamically weights implicit user-item interactions based on their relevance to the user's preferences.
- **Gating Mechanism**: Adaptively fuses explicit user embeddings with the attended implicit signal.

## Results

| Model       | RMSE      | Std       |
|-------------|-----------|-----------|
| ALS         | 0.856     | 1.22e-4   |
| SVD++       | 0.853     | 3.03e-4   |
| **SVDppAG** | **0.844** | **1.22e-4** |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train any of the implemented models:

```bash
python train_SVDppAG.py    # SVDppAG (proposed model)
python train_SVDpp.py      # SVD++ baseline
python train_ALS.py        # ALS baseline
```

Each script outputs the best validation RMSE and generates a submission file.

### Hyperparameter Search

```bash
python grid_search/grid_search_SVDppAG.py
python grid_search/grid_search_ALS.py
```

> SVD++ grid search can be performed using the SVDppAG script with attention and gating disabled in the config.

## Project Structure

```
├── configs/              # Model configuration files
├── data/                 # Dataset directory
│   ├── train_ratings.csv # Explicit ratings (sid, pid, rating)
│   ├── train_tbr.csv     # Implicit feedback (sid, pid)
│   └── sample_submission.csv
├── src/
│   ├── models/           # Model implementations
│   │   ├── SVDppAG.py
│   │   ├── SVDpp.py
│   │   └── ALS.py
│   ├── dataloader.py
│   └── eval.py
└── grid_search/          # Hyperparameter search scripts
```

## Models

| Model | Description |
|-------|-------------|
| **SVDppAG** | SVD++ enhanced with attention and gating mechanisms |
| SVD++ | Standard [SVD++](https://doi.org/10.1145/1401890.1401944) implementation |
| ALS | Alternating Least Squares for collaborative filtering |
