import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split as sklearn_split
import itertools
import pandas as pd
import time
from datetime import datetime

from src.dataloader import Dataloader as CSVLoader
from src.torch_dataset import SVDppDataset, svdpp_collate_fn
from src.torch_models import SVDppMLP, SVDpp
from src.torch_trainer import SVDppTrainer

RANDOM_STATE = 42

# loaders params
TEST_SIZE = 0.25
BATCH_SIZE = 1024
NUM_WORKERS = 2

# gs params
N_FACTORS_GRID = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
REG_LAMBDA_GRID = [0.01]
LEARNING_RATE = 0.003
NUM_EPOCHS = 15

# model to gs over
MODEL_CLASS = SVDpp # SVDppMLP

datetime_str = datetime.now().strftime("%Y-%m-%d")
RESULTS_FILE = f"experiments_data/{datetime_str}_gs_{type(MODEL_CLASS).__name__}.csv"

torch.manual_seed(RANDOM_STATE)
def run_grid_search():
    """Runs the grid search for hyperparameters."""

    print(f"--- Starting {type(MODEL_CLASS).__name__} Grid Search ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    print("Loading data...")
    ratings_df = CSVLoader.load_train_ratings()
    tbr_df = CSVLoader.load_train_tbr()
    ratings_train, ratings_valid = sklearn_split(ratings_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Training data size: {len(ratings_train)}, Validation data size: {len(ratings_valid)}")

    # build datasets and dataloaders
    print("Creating datasets and dataloaders...")
    train_dataset = SVDppDataset(ratings_train, tbr_df)
    valid_dataset = SVDppDataset(ratings_valid, tbr_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=svdpp_collate_fn, num_workers=NUM_WORKERS, pin_memory=True if device == 'cuda' else False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE*2, shuffle=False,
                              collate_fn=svdpp_collate_fn, num_workers=NUM_WORKERS, pin_memory=True if device == 'cuda' else False)

    n_users = train_dataset.num_scientists
    n_items = train_dataset.num_papers
    global_mean = train_dataset.global_mean
    print(f"Num users: {n_users}, Num items: {n_items}, Global mean: {global_mean:.4f}")

    # gs params
    best_rmse = float('inf')
    best_params = None
    results = []
    start_time = time.time()

    param_combinations = list(itertools.product(N_FACTORS_GRID, REG_LAMBDA_GRID))
    total_combinations = len(param_combinations)
    print(f"Starting grid search over {total_combinations} combinations...")

    for i, (n_factors, reg_lambda) in enumerate(param_combinations):
        print(f"\n--- Combination {i+1}/{total_combinations} ---")
        print(f"Parameters: n_factors={n_factors}, reg_lambda={reg_lambda}")
        combination_start_time = time.time()

        model = MODEL_CLASS(num_scientists=n_users, num_papers=n_items,
                            embedding_dim=n_factors, global_mean=global_mean).to(device)

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss(reduction='sum')

        trainer = SVDppTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            reg_lambda=reg_lambda,
            device=device,
            train_loader=train_loader,
            valid_loader=valid_loader,
            verbose=False
        )

        # train
        print(f"Training for {NUM_EPOCHS} epochs...")
        try:
            valid_rmse = trainer.train(num_epochs=NUM_EPOCHS)
            print(f"Finished Training. Best Validation RMSE: {valid_rmse:.4f}")

            results.append({
                'n_factors': n_factors,
                'reg_lambda': reg_lambda,
                'learning_rate': LEARNING_RATE,
                'num_epochs': NUM_EPOCHS,
                'best_valid_rmse': valid_rmse
            })

            if valid_rmse < best_rmse:
                best_rmse = valid_rmse
                best_params = {'n_factors': n_factors, 'reg_lambda': reg_lambda}
                print(f"*** New best RMSE found: {best_rmse:.4f} ***")

        except Exception as e:
            print(f"!!! Error during training for combination {n_factors}, {reg_lambda}: {e}")
            results.append({
                'n_factors': n_factors,
                'reg_lambda': reg_lambda,
                'learning_rate': LEARNING_RATE,
                'num_epochs': NUM_EPOCHS,
                'best_valid_rmse': float('nan')
            })

        combination_time = time.time() - combination_start_time
        print(f"Time for this combination: {combination_time:.2f} seconds")

    # print results
    total_time = time.time() - start_time
    print("\n--- Grid Search Finished ---")
    print(f"Total time: {total_time:.2f} seconds")

    if best_params:
        print(f"Best Validation RMSE: {best_rmse:.4f}")
        print(f"Best Parameters: {best_params}")
    else:
        print("Grid search did not find a successful combination.")

    # save
    if results:
        results_df = pd.DataFrame(results)
        try:
            results_df.to_csv(RESULTS_FILE, index=False)
            print(f"Grid search results saved to {RESULTS_FILE}")
        except IOError as e:
            print(f"Error saving results to CSV: {e}")
        print("\nTop 5 Results:")
        print(results_df.sort_values(by='best_valid_rmse').head())

if __name__ == "__main__":
    run_grid_search()