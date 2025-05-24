import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from src.dataloader import Dataloader
from src.torch_models import ALS
import src.ALS_helpers as ALS_helpers
import pandas as pd

RESULTS_FILE = "gs_results/ALS_grid_search.csv"

LAMBDAS = [0.5, 1.0, 2.0, 5.0, 10.0]
RANKS = [2, 4, 8, 16, 32, 64]
ITERATIONS = [30]
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
    


def main():
    # Load data
    print(" > Reading data...")
    ratings_df = Dataloader.load_train_ratings()
    ratings_train_valid, ratings_test = train_test_split(ratings_df, test_size=0.1, random_state=42)
    ratings_train, ratings_valid = train_test_split(ratings_train_valid, test_size=0.1/0.9, random_state=42)
    init_train_mat = torch.tensor(ratings_train.pivot(index="sid", columns="pid", values="rating").values, dtype=torch.float32)
    print(" > Data read completed")

    if normalize_rows:
        train_mat, mean, std = ALS_helpers.center_and_normalize_rows(init_train_mat)
        print("(Normalized and centered each column)")
    elif normalize_columns:
        train_mat, mean, std = ALS_helpers.center_and_normalize_columns(init_train_mat)
        print("(Normalized and centered each row)")
    else:
        train_mat = init_train_mat


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


    # Retrain model with best parameters on ratings_train + ratings_valid and evaluate on ratings_test
    train_valid_mat = torch.tensor(ratings_train_valid.pivot(index="sid", columns="pid", values="rating").values, dtype=torch.float32)
    if normalize_rows:
        train_valid_mat, mean, std = ALS_helpers.center_and_normalize_rows(train_valid_mat)
        print("(Normalized and centered each column)")
    elif normalize_columns:
        train_valid_mat, mean, std = ALS_helpers.center_and_normalize_columns(train_valid_mat)
        print("(Normalized and centered each row)")

    model = ALS(
        device="cuda" if torch.cuda.is_available() else "cpu",
        train_mat=train_valid_mat,
        tbr_df=None
    )
    test_loss = evaluate_model(model, best_params["lambda"], best_params["rank"], best_params["iterations"], mean, std, ratings_test)
    print(f" >>> LOSS WITH {best_params} ON TEST DATA: {test_loss}")

    # Save parameter configuration and losses to a CSV file
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

    print(" > END")



if __name__ == "__main__":
    main()

