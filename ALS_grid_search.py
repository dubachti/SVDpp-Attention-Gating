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


def main():
    # Load data
    print(" > READING DATA...")
    ratings_df = Dataloader.load_train_ratings()
    ratings_train, ratings_test = train_test_split(ratings_df, test_size=0.25, random_state=42)
    init_train_mat = torch.tensor(ratings_train.pivot(index="sid", columns="pid", values="rating").values, dtype=torch.float32)
    print(" > DATA READ")

    if normalize_rows:
        train_mat, mean, std = ALS_helpers.center_and_normalize_rows(init_train_mat)
        print(" > Normalized and centered each column")
    elif normalize_columns:
        train_mat, mean, std = ALS_helpers.center_and_normalize_columns(init_train_mat)
        print(" > Normalized and centered each row")
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
    best_params = ""
    results_list = []
    for lam in LAMBDAS:
        for rank in RANKS:
            for iterations in ITERATIONS:
                
                current_params = f"lambda={lam}, rank={rank}, iterations={iterations}"
                print(f"  - Parameters: {current_params}")
                model.train(lam=lam, rank=rank, iterations=iterations)
                predictions_matrix = model.get_predictions_matrix()

                if normalize_rows:
                    predictions_matrix = ALS_helpers.recover_matrix_rows(predictions_matrix, mean, std)
                elif normalize_columns:
                    predictions_matrix = ALS_helpers.recover_matrix_columns(predictions_matrix, mean, std)

                loss = ALS_helpers.evaluate_prediction_matrix(ratings_test, predictions_matrix)
                print(f"    => LOSS: {loss}")

                if loss < min_loss:
                    min_loss = loss
                    best_predictions_matrix = predictions_matrix
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


    # Create submission
    """print(" > CREATING SUBMISSION...")
    pred_fn = lambda sids, pids: best_predictions_matrix[sids, pids]
    ALS_helpers.make_submission(pred_fn, "ALS_submission.csv")"""

    print(" > END")



if __name__ == "__main__":
    main()

