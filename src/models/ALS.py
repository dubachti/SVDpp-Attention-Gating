import warnings
import numpy as np
import pandas as pd
from typing import Callable
import os
import torch


class ALS:
    """
    Alternating Least Squares implementation with support for NaN values. 
    Factorizes the ratings matrix A into two rank k matrices U and V such that A = U V^T.

    Parameters
    ----------
    device: torch.device
        The device (CPU or GPU) used for tensor computations.
    train_mat: torch.Tensor
        User-item rating matrix on which the model is trained on.
    """
    def __init__(self, device, train_mat):
        self.device = device
        self.train_mat = train_mat
        self.U = None
        self.V = None

    def predict_with_NaNs(self, train_mat, mean, std):

        # Random initialization of latent factors
        U = torch.randn(train_mat.shape[0], self.rank).to(self.device) * 0.01
        V = torch.randn(train_mat.shape[1], self.rank).to(self.device) * 0.01

        # Alternating optimization
        for i in range(self.iterations):
            U = self.optimize_U_with_NaNs(U, V, train_mat)
            V = self.optimize_V_with_NaNs(U, V, train_mat)

        self.U = U
        self.V = V
    
    def optimize_U_with_NaNs(self, U, V, A):
        assert U.shape[0] == A.shape[0] and U.shape[1] == V.shape[1] and V.shape[0] == A.shape[1]

        # Optimize matrix U while keeping V fixed
        n = U.shape[0]
        m = V.shape[0]
        rank = U.shape[1]

        A_masked = torch.nan_to_num(A, nan=0.0)
        B = V.T @ A_masked.T
        Id_lam = self.lam * torch.eye(rank, dtype=U.dtype, device=U.device)

        # Precompute of V_i^T V_i
        Q = V.unsqueeze(2) * V.unsqueeze(1)

        for j in range(n):
            
            mat = Id_lam.clone()

            valid_indices = (~torch.isnan(A[j, :])).nonzero(as_tuple=True)[0]

            if valid_indices.numel() > 0:
                mat += Q[valid_indices].sum(dim=0)

            U[j] = torch.linalg.solve(mat, B[:, j])

        return U
    
    def optimize_V_with_NaNs(self, U, V, A):
        assert U.shape[0] == A.shape[0] and U.shape[1] == V.shape[1] and V.shape[0] == A.shape[1]

        # Optimize matrix V while keeping U fixed
        n = U.shape[0]
        m = V.shape[0]
        rank = U.shape[1]

        A_masked = torch.nan_to_num(A, nan=0.0)
        B = U.T @ A_masked
        Id_lam = self.lam * torch.eye(rank, dtype=U.dtype, device=U.device)

        # Precompute of U_i^T U_i
        Q = U.unsqueeze(2) * U.unsqueeze(1)

        for j in range(m):
            mat = Id_lam.clone()

            valid_indices = (~torch.isnan(A[:, j])).nonzero(as_tuple=True)[0]
            if valid_indices.numel() > 0:
                mat += Q[valid_indices].sum(dim=0)

            V[j] = torch.linalg.solve(mat, B[:, j])

        return V
    
    def train(self, lam, rank, iterations):
        """
        Train the ALS model.

        Parameters
        ----------
        lam: float
            Regularization factor
        rank: int
            Rank of matrices U and V
        iterations: int
            Number of optimization iterations
        """
        self.lam = lam
        self.rank = rank
        self.iterations = iterations

        self.predict_with_NaNs(self.train_mat, None, None)

    def get_predictions_matrix(self):

        # Return the predicted ratings as a matrix
        return self.U @ self.V.T

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
    Recover the original matrix from a column-normalized matrix.
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
    df.to_csv(f"{filename}", index=False)