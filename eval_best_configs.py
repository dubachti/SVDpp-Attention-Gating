import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split as sklearn_split
import pandas as pd
import time
from datetime import datetime
import numpy as np
import os

from src.dataloader import Dataloader as CSVLoader
from src.torch_dataset import SVDppDataset, svdpp_collate_fn
from src.torch_models import SVDppAG
from src.torch_trainer import SVDppTrainer

# --- Configuration ---
RANDOM_STATE = 42
NEW_TRAINING_SEED = 42
MODEL_CLASS = SVDppAG

TEST_SIZE = 0.1
VALIDATION_SIZE = 0.1

BATCH_SIZE = 1024
NUM_WORKERS = 2

MAX_TRAINING_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA_IMPROVEMENT = 0.0001

# log file
GRID_SEARCH_RESULTS_FILE = "gs_results/2025-05-11-11-01-11_SVDppAG_grid_search.csv"
datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
TEST_RESULTS_DIR = "gs_results"
TEST_RESULTS_FILE = f"{TEST_RESULTS_DIR}/{datetime_str}_{MODEL_CLASS.__name__}_best_configs_test_results.csv"

# --- Global variables for data loaders and stats ---
g_train_loader = None
g_valid_loader = None
g_test_loader = None
g_n_users = None
g_n_items = None
g_global_mean = None
g_device = None

def load_and_prepare_final_data():
    global g_train_loader, g_valid_loader, g_test_loader
    global g_n_users, g_n_items, g_global_mean, g_device

    g_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {g_device}")

    print("Loading raw data...")
    ratings_df = CSVLoader.load_train_ratings()
    tbr_df = CSVLoader.load_train_tbr()
    
    # original split ratios, same random state
    ratings_temp, ratings_test = sklearn_split(ratings_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)    
    ratings_train, ratings_valid = sklearn_split(ratings_temp, test_size=VALIDATION_SIZE/(1-TEST_SIZE), random_state=RANDOM_STATE)

    print(f"Training data size: {len(ratings_train)}, Validation data size: {len(ratings_valid)}, Test data size: {len(ratings_test)}")

    
    # --- Create datasets and dataloaders ---
    train_dataset = SVDppDataset(ratings_train, tbr_df)
    valid_dataset = SVDppDataset(ratings_valid, tbr_df)
    test_dataset = SVDppDataset(ratings_test, tbr_df)

    # get global stats
    g_n_users = train_dataset.num_scientists
    g_n_items = train_dataset.num_papers
    g_global_mean = train_dataset.global_mean
    print(f"Num users: {g_n_users}, Num items: {g_n_items}, Global mean: {g_global_mean:.4f}")

    common_loader_args = {
        'collate_fn': svdpp_collate_fn,
        'num_workers': NUM_WORKERS,
        'pin_memory': True if g_device.type == 'cuda' else False,
        'persistent_workers': True if NUM_WORKERS > 0 else False
    }

    g_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **common_loader_args)
    g_valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE*2, shuffle=False, **common_loader_args)
    g_test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, **common_loader_args)


def evaluate_best_configurations():
    global g_train_loader, g_valid_loader, g_test_loader
    global g_n_users, g_n_items, g_global_mean, g_device

    print(f"--- Starting Evaluation of Best Configs from {GRID_SEARCH_RESULTS_FILE} ---")
    load_and_prepare_final_data()

    try:
        grid_results_df = pd.read_csv(GRID_SEARCH_RESULTS_FILE)
    except FileNotFoundError:
        print(f"Error: Grid search results file not found at '{GRID_SEARCH_RESULTS_FILE}'")
        return

    # ensure correct types for key columns from CSV
    grid_results_df['n_factors'] = pd.to_numeric(grid_results_df['n_factors'])
    grid_results_df['validation_rmse'] = pd.to_numeric(grid_results_df['validation_rmse'])

    # find best config for each n_factors
    best_configs_indices = grid_results_df.groupby('n_factors')['validation_rmse'].idxmin()
    best_configs_df = grid_results_df.loc[best_configs_indices]
    
    print(f"\nFound {len(best_configs_df)} best configurations to evaluate (one per n_factors):")
    print(best_configs_df[['n_factors', 'reg_lambda', 'learning_rate', 'use_attn', 'use_gating', 'validation_rmse']].to_string())

    test_evaluation_results = []
    overall_start_time = time.time()

    for _, config_row in best_configs_df.iterrows():
        n_factors = int(config_row['n_factors'])
        reg_lambda = float(config_row['reg_lambda'])

        use_attn_val = config_row['use_attn']
        use_attn = str(use_attn_val).lower() == 'true' if isinstance(use_attn_val, str) else bool(use_attn_val)
        use_gating_val = config_row['use_gating']
        use_gating = str(use_gating_val).lower() == 'true' if isinstance(use_gating_val, str) else bool(use_gating_val)
        learning_rate = float(config_row['learning_rate'])
        
        current_params_str = (f"n_factors={n_factors}, reg_lambda={reg_lambda}, lr={learning_rate}, "
                              f"use_attn={use_attn}, use_gating={use_gating}")
        print(f"\n--- Evaluating: {current_params_str} ---")

        torch.manual_seed(NEW_TRAINING_SEED)
        np.random.seed(NEW_TRAINING_SEED)
        if g_device.type == 'cuda':
            torch.cuda.manual_seed_all(NEW_TRAINING_SEED)
        
        trial_start_time = time.time()
        
        try:
            model = MODEL_CLASS(
                num_scientists=g_n_users,
                num_papers=g_n_items,
                embedding_dim=n_factors,
                global_mean=g_global_mean,
                use_attention=use_attn,
                use_gating=use_gating,
            ).to(g_device)

            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss(reduction='sum') 
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

            trainer = SVDppTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                reg_lambda=reg_lambda,
                device=g_device,
                train_loader=g_train_loader,
                valid_loader=g_valid_loader,
                scheduler=scheduler,
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                min_delta=MIN_DELTA_IMPROVEMENT,
                verbose=False
            )

            print(f"Training for up to {MAX_TRAINING_EPOCHS} epochs...")
            best_train_val_rmse = trainer.train(num_epochs=MAX_TRAINING_EPOCHS)
            
            epochs_completed = trainer.best_valid_rmse_epoch + 1

            print(f"Finished training. Best RMSE on internal validation set: {best_train_val_rmse:.4f} (Achieved at epoch {epochs_completed})")

            print("Evaluating on the final test set...")
            test_rmse = trainer.evaluate_on_loader(g_test_loader)
            print(f"Test RMSE: {test_rmse:.4f}")

            result_entry = {
                'n_factors': n_factors,
                'reg_lambda': reg_lambda,
                'learning_rate': learning_rate,
                'use_attn': use_attn,
                'use_gating': use_gating,
                'original_grid_val_rmse': config_row['validation_rmse'],
                'train_phase_val_rmse': best_train_val_rmse,
                'train_phase_epochs': epochs_completed,
                'final_test_rmse': test_rmse
            }
            test_evaluation_results.append(result_entry)

        except Exception as e:
            print(f"!!! Error during evaluation for params {current_params_str}: {e}")
            import traceback
            traceback.print_exc()
            test_evaluation_results.append({
                'n_factors': n_factors, 'reg_lambda': reg_lambda, 'learning_rate': learning_rate,
                'use_attn': use_attn, 'use_gating': use_gating,
                'original_grid_val_rmse': config_row['validation_rmse'],
                'error_message': str(e)
            })

        trial_duration = time.time() - trial_start_time
        print(f"Time for this configuration: {trial_duration:.2f} seconds")

    overall_duration = time.time() - overall_start_time
    print(f"\n--- Evaluation Finished ---")
    print(f"Total evaluation time: {overall_duration:.2f} seconds")

    results_df = pd.DataFrame(test_evaluation_results)
    try:
        os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
        results_df.to_csv(TEST_RESULTS_FILE, index=False, float_format='%.6f')
        print(f"\nTest evaluation results saved to {TEST_RESULTS_FILE}")
        print("\nResults Summary (Sorted by Final Test RMSE):")
        print(results_df.sort_values(by='final_test_rmse').to_string())
    except IOError as e:
        print(f"Error saving results to CSV: {e}")
    except Exception as e:
        print(f"An error occurred while processing results: {e}")

if __name__ == "__main__":
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    evaluate_best_configurations()