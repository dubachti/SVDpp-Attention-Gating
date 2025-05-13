import warnings
import numpy as np
import pandas as pd
from typing import Callable
import os




# DATA CENTERING AND NORMALIZATION
# column-wise
def center_and_normalize_columns(matrix):
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    normalized_matrix = (matrix - mean) / std
    return normalized_matrix, mean, std

def recover_matrix_columns(normalized_matrix, mean, std):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        mean = mean.reshape(1, -1)
        std = std.reshape(1, -1)
        return normalized_matrix * std + mean

# row-wise
def center_and_normalize_rows(matrix):
    mean = np.nanmean(matrix, axis=1, keepdims=True)
    std = np.nanstd(matrix, axis=1, keepdims=True)
    normalized_matrix = (matrix - mean) / std
    return normalized_matrix, mean.squeeze(), std.squeeze()

def recover_matrix_rows(normalized_matrix, mean, std):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        mean = mean.reshape(-1, 1)
        std = std.reshape(-1, 1)
        return normalized_matrix * std + mean
    

# MODEL EVALUATION
def evaluate(valid_df: pd.DataFrame, pred_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> float:
    from sklearn.metrics import root_mean_squared_error
    preds = pred_fn(valid_df["sid"].values, valid_df["pid"].values)
    return root_mean_squared_error(valid_df["rating"].values, preds)

def evaluate_prediction_matrix(valid_df, prediction_matrix):
    pred_fn = lambda sids, pids: prediction_matrix[sids, pids]
    val_score = evaluate(valid_df, pred_fn)
    return val_score


# CREATE SUBMISSION
def make_submission(pred_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], filename: os.PathLike):

    df = pd.read_csv(os.path.join("data", "sample_submission.csv"))

    # Get sids and pids
    sid_pid = df["sid_pid"].str.split("_", expand=True)
    sids = sid_pid[0]
    pids = sid_pid[1]
    sids = sids.astype(int).values
    pids = pids.astype(int).values

    df["rating"] = pred_fn(sids, pids)
    df.to_csv(f"experiments_data/{filename}", index=False)