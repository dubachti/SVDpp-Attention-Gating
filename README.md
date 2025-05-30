# SVDppAG: Enhancing SVD++ with Attention and Gating

### Overview
This repository implements SVDppAG, an enhanced version of the SVD++ collaborative filtering model. SVD++ is a well-regarded matrix factorization technique that leverages both explicit user ratings and implicit user feedback for recommendations. However, its standard approach to aggregating implicit feedback treats all such interactions equally, and the fusion of this implicit signal with the explicit user preference is a simple summation.

SVDppAG addresses these limitations by introducing two key mechanisms:
1.  **Attention Mechanism**: This allows the model to dynamically weight implicit user-item interactions based on their relevance to the user's preferences, enabling a more focused aggregation of the implicit signal.
2.  **Gating Mechanism**: This provides an adaptive way to fuse the explicit user signal with the attended implicit item signal, allowing the model to learn the optimal influence of each component.

The goal of SVDppAG is to improve prediction accuracy in collaborative filtering tasks by enabling the model to better understand the nuanced relationships within user-item interactions.

### Setup
```sh
pip install -r requirements.txt
```

### Models
- **SVDppAG**: SVD++ model enhanced with attention and gating mechanisms.

Some additional models are also included for comparison:
- **ALS**: Alternating Least Squares implementation for collaborative filtering.
- **SVD++**: implementation of the [SVD++](https://doi.org/10.1145/1401890.1401944) model.


### Training
Individual models can be trained using the following scripts:
- **SVDppAG**:
```sh
python train_SVDppAG.py
```
- **SVD++**:
```sh
python train_SVDpp.py
```
- **ALS**:
```sh
python train_ALS.py
```
Each of the scripts will print the best achieved RMSE on the validation set and create a submission file for the competition.

### Hyperparameter Tuning
Grid search for hyperparameter optimization can be performed using:
- **SVDppAG**:
```sh
python grid_search/grid_search_SVDppAG.py
```
- **ALS**:
```sh
python grid_search/grid_search_ALS.py
```

Grid search on **SVDpp++**  can be performed using the **SVDppAG** script with deactivated attention and gating.

### Dataset
This project uses a dataset consisting of explicit user-item ratings and implicit user feedback. The scripts expect the following files in the `data/` directory:
- `train_ratings.csv`: Contains explicit ratings with columns `sid`, `pid`, `rating`.
- `train_tbr.csv`: Contains implicit feedback with columns `sid`, `pid`.
- `sample_submission.csv`: Used for generating prediction files for the competition.

### Results
Below we provide a summary of the results obtained from the models on the test set:
| Model     | RMSE   | Std     |
|-----------|--------|---------|
| ALS       | 0.856  | 1.22e-4 |
| SVD++     | 0.853  | 3.03e-4 |
| **SVDppAG**   | **0.844**  | **1.22e-4** |
