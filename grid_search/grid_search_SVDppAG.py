"""
Grid Search for SVD++ with Attention and Gating Mechanisms

Stores results to a CSV file in the 'grid_search_results' directory.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split as sklearn_split
import pandas as pd
import time
from datetime import datetime
import numpy as np
import itertools
import os
import sys

# add project root to sys.path to import local modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataloader import Dataloader as CSVLoader
from src.SVDpp_dataset import SVDppDataset, svdpp_collate_fn
from src.models.SVDpp import SVDpp
from src.models.SVDppAG import SVDppAG
from src.SVDpp_trainer import SVDppTrainer

# config
RANDOM_STATE = 42
MODEL_CLASS = SVDppAG

# data loading
TEST_SIZE = 0.1
VALIDATION_SIZE = 0.1
BATCH_SIZE = 1024
NUM_WORKERS = 2

# early stopping
MAX_TRAINING_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA_IMPROVEMENT = 0.0001

# grid search hyperparams
SEARCH_SPACE_GRID = {
    'n_factors': [4, 8, 16, 32, 64, 128, 256],
    'reg_lambda': [0.005, 0.01, 0.02, 0.04],
    'use_attn': [True], # used for ablation study
    'use_gating': [True], # used for ablation study
    'learning_rate': [0.0001, 0.001, 0.01],
}

datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
RESULTS_DIR = "grid_search_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = f"{RESULTS_DIR}/{datetime_str}_{MODEL_CLASS.__name__}_grid_search_results.csv"

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# globals to not load them again on every train run
g_train_loader = None
g_valid_loader = None
g_n_users = None
g_n_items = None
g_global_mean = None
g_device = None

def load_and_prepare_data():
    global g_train_loader, g_valid_loader, g_n_users, g_n_items, g_global_mean, g_device

    g_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {g_device}")

    print("Loading data...")
    ratings_df = CSVLoader.load_train_ratings()
    tbr_df = CSVLoader.load_train_tbr()

    ratings_temp, ratings_test = sklearn_split(ratings_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)    
    ratings_train, ratings_valid = sklearn_split(ratings_temp, test_size=VALIDATION_SIZE/(1-TEST_SIZE), random_state=RANDOM_STATE)
    
    print(f"Training data size: {len(ratings_train)}, Validation data size: {len(ratings_valid)}, Test data size: {len(ratings_test)}")

    print("Creating datasets and dataloaders...")
    train_dataset = SVDppDataset(ratings_train, tbr_df)
    valid_dataset = SVDppDataset(ratings_valid, tbr_df)

    g_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                collate_fn=svdpp_collate_fn, num_workers=NUM_WORKERS,
                                pin_memory=True if g_device.type == 'cuda' else False,
                                persistent_workers=True if NUM_WORKERS > 0 else False)
    g_valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE*2, shuffle=False,
                                collate_fn=svdpp_collate_fn, num_workers=NUM_WORKERS,
                                pin_memory=True if g_device.type == 'cuda' else False,
                                persistent_workers=True if NUM_WORKERS > 0 else False)

    g_n_users = train_dataset.num_scientists
    g_n_items = train_dataset.num_papers
    g_global_mean = train_dataset.global_mean
    print(f"Num users: {g_n_users}, Num items: {g_n_items}, Global mean: {g_global_mean:.4f}")

def objective_function(n_factors, reg_lambda, use_attn, use_gating, learning_rate):
    global g_train_loader, g_valid_loader, g_n_users, g_n_items, g_global_mean, g_device
    
    current_params = {
        'n_factors': n_factors, 'reg_lambda': reg_lambda, 'use_attn': use_attn,
        'use_gating': use_gating, 'learning_rate': learning_rate,
    }
    print(f"\n--- Evaluating Parameters ---")
    print(current_params)
    
    trial_start_time = time.time()
    
    try:
        model = MODEL_CLASS(
            num_scientists=g_n_users,
            num_papers=g_n_items,
            embedding_dim=n_factors,
            global_mean=g_global_mean,
            use_attention=use_attn,
            use_gating=use_gating
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
        best_val_rmse = trainer.train(num_epochs=MAX_TRAINING_EPOCHS) 
        
        print(f"Finished Training. Best Validation RMSE: {best_val_rmse:.4f}")

    except Exception as e:
        print(f"!!! Error during training with params {current_params}: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')

    trial_duration = time.time() - trial_start_time
    print(f"Time for this evaluation: {trial_duration:.2f} seconds")
    
    return best_val_rmse

def run_grid_search():
    """Grid search for hyperparameter tuning."""
    print(f"--- Starting {MODEL_CLASS.__name__} Grid Search ---")
    
    load_and_prepare_data()

    start_time = time.time()
    
    param_names = list(SEARCH_SPACE_GRID.keys())
    param_value_lists = list(SEARCH_SPACE_GRID.values())
    
    all_combinations = list(itertools.product(*param_value_lists))
    total_combinations = len(all_combinations)
    print(f"Total parameter combinations to evaluate: {total_combinations}")

    results_data = []
    best_overall_rmse = float('inf')
    best_params = {}

    for i, param_combination in enumerate(all_combinations):
        current_run_params = {name: val for name, val in zip(param_names, param_combination)}
        print(f"\n--- Grid Search Iteration {i+1}/{total_combinations} ---")
        
        validation_rmse = objective_function(
            n_factors=current_run_params['n_factors'],
            reg_lambda=current_run_params['reg_lambda'],
            use_attn=current_run_params['use_attn'],
            use_gating=current_run_params['use_gating'],
            learning_rate=current_run_params['learning_rate']
        )
        
        trial_result = current_run_params.copy()
        trial_result['validation_rmse'] = validation_rmse
        results_data.append(trial_result)

        if validation_rmse < best_overall_rmse:
            best_overall_rmse = validation_rmse
            best_params = current_run_params
            print(f"!!! New best RMSE found: {best_overall_rmse:.4f} with params: {best_params}")

    total_time = time.time() - start_time
    print("\n--- Grid Search Finished ---")
    print(f"Total search time: {total_time:.2f} seconds")

    print(f"Best Validation RMSE: {best_overall_rmse:.4f}")
    print("Best Parameters:")
    for name, val in best_params.items():
        print(f"  {name}: {val}")

    # save results to CSV
    results_df = pd.DataFrame(results_data)
    try:
        results_df.to_csv(RESULTS_FILE, index=False)
        print(f"\nGrid search results saved to {RESULTS_FILE}")
        print("\nTop 5 Trials (Sorted by RMSE):")
        print(results_df.sort_values(by='validation_rmse').head())
    except IOError as e:
        print(f"Error saving results to CSV: {e}")
    except Exception as e:
        print(f"An error occurred while processing results: {e}")

if __name__ == "__main__":
    run_grid_search()