import warnings
import numpy as np
import pandas as pd
from typing import Callable
import os


def center_and_normalize_columns(matrix):
    """
    Normalize the matrix column-wise by subtracting the mean and dividing by the standard deviation.
    """
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    normalized_matrix = (matrix - mean) / std
    return normalized_matrix, mean, std

def recover_matrix_columns(normalized_matrix, mean, std):
    """
    
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        mean = mean.reshape(1, -1)
        std = std.reshape(1, -1)
        return normalized_matrix * std + mean

def center_and_normalize_rows(matrix):
    """
    Normalize the matrix column-wise by subtracting the mean and dividing by the standard deviation.
    """
    mean = np.nanmean(matrix, axis=1, keepdims=True)
    std = np.nanstd(matrix, axis=1, keepdims=True)
    normalized_matrix = (matrix - mean) / std
    return normalized_matrix, mean.squeeze(), std.squeeze()

def recover_matrix_rows(normalized_matrix, mean, std):
    """
    Recover the original matrix from a row-normalized matrix.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        mean = mean.reshape(-1, 1)
        std = std.reshape(-1, 1)
        return normalized_matrix * std + mean
    

def evaluate(valid_df: pd.DataFrame, pred_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> float:
    """
    Evaluate model predictions using Root Mean Squared Error (RMSE).
    """
    from sklearn.metrics import root_mean_squared_error
    preds = pred_fn(valid_df["sid"].values, valid_df["pid"].values)
    return root_mean_squared_error(valid_df["rating"].values, preds)

def evaluate_prediction_matrix(valid_df, prediction_matrix):
    """
    Compare predictions and validation data.
    """
    pred_fn = lambda sids, pids: prediction_matrix[sids, pids]
    val_score = evaluate(valid_df, pred_fn)
    return val_score


def make_submission(pred_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], filename: os.PathLike):
    """
    Predict all ratings and save the result to a CSV file.
    """
    df = pd.read_csv(os.path.join("data", "sample_submission.csv"))

    sid_pid = df["sid_pid"].str.split("_", expand=True)
    sids = sid_pid[0]
    pids = sid_pid[1]
    sids = sids.astype(int).values
    pids = pids.astype(int).values

    df["rating"] = pred_fn(sids, pids)
    df.to_csv(f"experiments_data/{filename}", index=False)
