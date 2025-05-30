# SVDppAG: Enhancing SVD++ with Attention and Gating

### Overview
This repository implements SVDppAG, an enhanced version of the SVD++ collaborative filtering model. SVD++ is a well-regarded matrix factorization technique that leverages both explicit user ratings and implicit user feedback for recommendations. However, its standard approach to aggregating implicit feedback treats all such interactions equally, and the fusion of this implicit signal with the explicit user preference is a simple summation.

SVDppAG addresses these limitations by introducing two key mechanisms:
1.  **Attention Mechanism**: This allows the model to dynamically weight implicit user-item interactions based on their relevance to the user's preferences, enabling a more focused aggregation of the implicit signal.
2.  **Gating Mechanism**: This provides an adaptive way to fuse the explicit user signal with the attended implicit item signal, allowing the model to learn the optimal influence of each component.

The goal of SVDppAG is to improve prediction accuracy in collaborative filtering tasks by enabling the model to better understand the nuanced relationships within user-item interactions.

### Setup
```sh
git clone git@github.com:dubachti/SVDpp-Attention-Gating.git
cd SVDpp-Attention-Gating
pip install -r requirements.txt
```

### Models
- **ALS**: Alternating Least Squares implementation for collaborative filtering.
- **SVD++**: implementation of the [SVD++](https://doi.org/10.1145/1401890.1401944) model.
- **SVDppAG**: SVD++ model enhanced with attention and gating mechanisms.


### Training
Individual models can be trained using the following scripts:
- **ALS**:
```sh
python train_ALS.py
```
- **SVD++**:
```sh
python train_SVDpp.py
```
- **SVDppAG**:
```sh
python train_SVDppAG.py
```

### Hyperparameter Tuning
Grid search for hyperparameter optimization can be performed using:
- **ALS**:
```sh
python grid_search_ALS.py
```
- **SVDppAG**:
```sh
python grid_search_SVDppAG.py
```

Grid search on **SVDpp++**  can be performed using the **SVDppAG** script with deactivated attention and gating.

### Dataset
This project uses a dataset consisting of explicit user-item ratings and implicit user feedback. The scripts expect the following files in the `data/` directory:
- `train_ratings.csv`: Contains explicit ratings with columns like `sid`, `pid`, `rating`.
- `train_tbr.csv`: Contains implicit feedback with columns like `sid`, `pid`.
- `sample_submission.csv`: Used for generating prediction files.
