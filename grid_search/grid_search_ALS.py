import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
# add project root to sys.path to import local modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.dataloader import Dataloader
from src.models.ALS import *

RANDOM_STATE = 42
TEST_SIZE = 0.1
VALIDATION_SIZE = 0.1

# Parameters over which to perform grid search
LAMBDAS = [16, 18, 20, 22, 24]
RANKS = [4, 8, 12, 16, 20, 24]
ITERATIONS = [50]


datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
RESULTS_DIR = "grid_search_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = f"{RESULTS_DIR}/{datetime_str}_ALS_grid_search_results.csv"

# Enable normalization along either columns, rows, or none at all
normalize_rows = False
normalize_columns = True

def evaluate_model(model, lam, rank, iterations, mean, std, ratings_valid):
    
    # Train model and get predictions
    model.train(lam=lam, rank=rank, iterations=iterations)
    predictions_matrix = model.get_predictions_matrix()

    # Recover initial ratings by reverting the normalization
    if normalize_rows:
        predictions_matrix = recover_matrix_rows(predictions_matrix, mean, std)
    elif normalize_columns:
        predictions_matrix = recover_matrix_columns(predictions_matrix, mean, std)

    # Compute and return the loss of the current model
    loss = evaluate_prediction_matrix(ratings_valid, predictions_matrix)
    return loss
    
def grid_search():
    # Load data
    print(" > Reading data...")
    ratings_df = Dataloader.load_train_ratings()
    ratings_temp, _ = train_test_split(ratings_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)    
    ratings_train, ratings_valid = train_test_split(ratings_temp, test_size=VALIDATION_SIZE/(1-TEST_SIZE), random_state=RANDOM_STATE)
    init_train_mat = torch.tensor(ratings_train.pivot(index="sid", columns="pid", values="rating").values, dtype=torch.float32)
    print(" > Data read completed")

    # Normalize and center data if enabled
    if normalize_rows:
        train_mat, mean, std = center_and_normalize_rows(init_train_mat)
        print("(Normalized and centered each row)")
    elif normalize_columns:
        train_mat, mean, std = center_and_normalize_columns(init_train_mat)
        print("(Normalized and centered each column)")
    else:
        train_mat, mean, std = init_train_mat, 0, 0

    # Initialize model
    model = ALS(
        device="cuda" if torch.cuda.is_available() else "cpu",
        train_mat=train_mat
    )

    # Grid search over model parameters
    print(" > STARTING GRID SEARCH...")
    min_loss = float("inf")
    best_params = {}
    results_list = []
    for lam in LAMBDAS:
        for rank in RANKS:
            for iterations in ITERATIONS:
                
                current_params = {}
                current_params["lambda"] = lam
                current_params["rank"] = rank
                current_params["iterations"] = iterations
                print(f"  - Parameters: {current_params}")

                loss = evaluate_model(model, lam, rank, iterations, mean, std, ratings_valid)
                print(f"    => LOSS: {loss}")
                if loss < min_loss:
                    min_loss = loss
                    best_params = current_params

                params_dict = {}
                params_dict["lambda"] = lam
                params_dict["rank"] = rank
                params_dict["iterations"] = iterations
                params_dict["loss"] = loss
                results_list.append(params_dict)

    print(f" > GRID SEARCH END: best parameters are {best_params} with loss={min_loss}")

    # Save parameter configuration and losses to a CSV file
    results_df = pd.DataFrame(results_list)
    try:
        results_df.to_csv(RESULTS_FILE, index=False)
        print(f"\nGrid search results saved to {RESULTS_FILE}")
    except IOError as e:
        print(f"Error saving results to CSV: {e}")
    except Exception as e:
        print(f"An error occurred while processing results: {e}")

    print(" > GRID SEARCH END")

    return best_params

def n_runs_with_best_params(best_params, n_runs=3):

    # Read data
    print(" > Reading data...")
    ratings_df = Dataloader.load_train_ratings()

    ratings_temp, ratings_test = train_test_split(ratings_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)    
    ratings_train, _ = train_test_split(ratings_temp, test_size=VALIDATION_SIZE/(1-TEST_SIZE), random_state=RANDOM_STATE)
    
    init_train_mat = torch.tensor(ratings_train.pivot(index="sid", columns="pid", values="rating").values, dtype=torch.float32)
    print(" > Data read completed")

    # Normalize and center data if enabled
    if normalize_rows:
        train_mat, mean, std = center_and_normalize_rows(init_train_mat)
        print("(Normalized and centered each row)")
    elif normalize_columns:
        train_mat, mean, std = center_and_normalize_columns(init_train_mat)
        print("(Normalized and centered each column)")
    else:
        train_mat, mean, std = init_train_mat, 0, 0

    # Train (on init_train_mat) and evaluate (on ratings_test) three times with new seed each time
    losses = []
    for i in range(n_runs):
        seed = torch.randint(1, 100000, (1,))
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = ALS(
            device="cuda" if torch.cuda.is_available() else "cpu",
            train_mat=train_mat
        )
        test_loss = evaluate_model(model, best_params["lambda"], best_params["rank"], best_params["iterations"], mean, std, ratings_test)
        losses.append(test_loss)
        print(f" > LOSS WITH {best_params} ON TEST DATA: {test_loss}  [seed={seed}]")

    # Print the average loss and standard deviation over all trained models
    print(f" >>> LOSS OVER {n_runs} RUNS: mean={np.mean(losses)}, std={np.std(losses)}")

if __name__ == "__main__":

    # Run grid search over the pre-defined parameter space
    best_params = grid_search()

    # Use the best parameters from grid search and evaluate the model
    n_runs_with_best_params(best_params)
