import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from src.dataloader import Dataloader
from src.torch_models import ALS
import src.ALS_helpers as ALS_helpers
import pandas as pd
import numpy as np

RESULTS_FILE = "gs_results/ALS_grid_search.csv"
save_to_file = False

# These grid search parameters are based on a more coarse grid search done beforehand
LAMBDAS = [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
RANKS = [12, 13, 14, 15, 16]
ITERATIONS = [50]
normalize_rows = False
normalize_columns = True


def evaluate_model(model, lam, rank, iterations, mean, std, ratings_valid):
    
    model.train(lam=lam, rank=rank, iterations=iterations)
    predictions_matrix = model.get_predictions_matrix()

    if normalize_rows:
        predictions_matrix = ALS_helpers.recover_matrix_rows(predictions_matrix, mean, std)
    elif normalize_columns:
        predictions_matrix = ALS_helpers.recover_matrix_columns(predictions_matrix, mean, std)

    loss = ALS_helpers.evaluate_prediction_matrix(ratings_valid, predictions_matrix)
    return loss
    


def grid_search():

    # Load data
    print(" > Reading data...")
    ratings_df = Dataloader.load_train_ratings()
    ratings_train_valid, _ = train_test_split(ratings_df, test_size=0.1, random_state=42)
    ratings_train, ratings_valid = train_test_split(ratings_train_valid, test_size=0.1/0.9, random_state=42)
    init_train_mat = torch.tensor(ratings_train.pivot(index="sid", columns="pid", values="rating").values, dtype=torch.float32)
    print(" > Data read completed")

    
    if normalize_rows:
        train_mat, mean, std = ALS_helpers.center_and_normalize_rows(init_train_mat)
        print("(Normalized and centered each row)")
    elif normalize_columns:
        train_mat, mean, std = ALS_helpers.center_and_normalize_columns(init_train_mat)
        print("(Normalized and centered each column)")
    else:
        train_mat, mean, std = init_train_mat, 0, 0


    # Initialize model, TODO: use tbr list
    model = ALS(
        device="cuda" if torch.cuda.is_available() else "cpu",
        train_mat=train_mat,
        tbr_df=None
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
    if save_to_file:
        results_df = pd.DataFrame(results_list)
        try:
            results_df.to_csv(RESULTS_FILE, index=False)
            print(f"\nGrid search results saved to {RESULTS_FILE}")
        except IOError as e:
            print(f"Error saving results to CSV: {e}")
        except Exception as e:
            print(f"An error occurred while processing results: {e}")


    # Create submission
    """print(" > CREATING SUBMISSION...")
    pred_fn = lambda sids, pids: best_predictions_matrix[sids, pids]
    ALS_helpers.make_submission(pred_fn, "ALS_submission.csv")"""

    print(" > GRID SEARCH END")

    return best_params


def three_runs_with_best_params(best_params):

    # Read data
    print(" > Reading data...")
    ratings_df = Dataloader.load_train_ratings()
    ratings_train_valid, ratings_test = train_test_split(ratings_df, test_size=0.1, random_state=42)
    ratings_train, ratings_valid = train_test_split(ratings_train_valid, test_size=0.1/0.9, random_state=42)
    init_train_mat = torch.tensor(ratings_train.pivot(index="sid", columns="pid", values="rating").values, dtype=torch.float32)
    print(" > Data read completed")


    # Normalize data
    if normalize_rows:
        train_mat, mean, std = ALS_helpers.center_and_normalize_rows(init_train_mat)
        print("(Normalized and centered each row)")
    elif normalize_columns:
        train_mat, mean, std = ALS_helpers.center_and_normalize_columns(init_train_mat)
        print("(Normalized and centered each column)")
    else:
        train_mat, mean, std = init_train_mat, 0, 0


    # Train (on init_train_mat) and evaluate (on ratings_test) three times with new seed each time
    nof_runs = 3
    losses = []
    for i in range(nof_runs):
        
        seed = torch.randint(1, 100000, (1,))
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = ALS(
            device="cuda" if torch.cuda.is_available() else "cpu",
            train_mat=train_mat,
            tbr_df=None
        )
        test_loss = evaluate_model(model, best_params["lambda"], best_params["rank"], best_params["iterations"], mean, std, ratings_test)
        losses.append(test_loss)
        print(f" > LOSS WITH {best_params} ON TEST DATA: {test_loss}  [seed={seed}]")

    print(f" >>> LOSS OVER {nof_runs} RUNS: mean={np.mean(losses)}, std={np.std(losses)}")


if __name__ == "__main__":
    best_params = grid_search()
    three_runs_with_best_params(best_params)
