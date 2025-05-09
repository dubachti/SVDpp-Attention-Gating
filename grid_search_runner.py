import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split as sklearn_split
import pandas as pd
import time
from datetime import datetime
import numpy as np

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from src.dataloader import Dataloader as CSVLoader
from src.torch_dataset import SVDppDataset, svdpp_collate_fn
from src.torch_models import SVDppAG
from src.torch_trainer import SVDppTrainer

# config
RANDOM_STATE = 42
MODEL_CLASS = SVDppAG

# data loading
TEST_SIZE = 0.25
BATCH_SIZE = 1024
NUM_WORKERS = 2

# early stopping
MAX_TRAINING_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA_IMPROVEMENT = 0.0001

# grid search hyperparams
N_CALLS = 20
N_INITIAL_POINTS = 5 

# search space
SEARCH_SPACE = [
    Integer(1, 300, name='n_factors'),
    Real(1e-5, 1e-1, prior='log-uniform', name='reg_lambda'),
    Categorical([False], name='use_attn'),
    Categorical([False], name='use_gating'),
    Real(1e-4, 1e-1, prior='log-uniform', name='learning_rate'),
]

datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
RESULTS_FILE = f"experiments_data/{datetime_str}_{MODEL_CLASS.__name__}.csv"

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

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
    ratings_train, ratings_valid = sklearn_split(ratings_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Training data size: {len(ratings_train)}, Validation data size: {len(ratings_valid)}")

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

@use_named_args(dimensions=SEARCH_SPACE)
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

        trainer = SVDppTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            reg_lambda=reg_lambda,
            device=g_device,
            train_loader=g_train_loader,
            valid_loader=g_valid_loader,
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

def run_bayesian_optimization():
    print(f"--- Starting {MODEL_CLASS.__name__} Bayesian Optimization ---")
    
    load_and_prepare_data()

    start_time = time.time()
    
    # gp_minimize optimization using Gaussian Processes
    results = gp_minimize(
        func=objective_function,
        dimensions=SEARCH_SPACE,
        n_calls=N_CALLS,
        n_initial_points=N_INITIAL_POINTS,
        random_state=RANDOM_STATE,
        verbose=True
    )

    total_time = time.time() - start_time
    print("\n--- Bayesian Optimization Finished ---")
    print(f"Total optimization time: {total_time:.2f} seconds")

    print(f"Best Validation RMSE: {results.fun:.4f}")
    print("Best Parameters:")
    best_params_names = [dim.name for dim in SEARCH_SPACE]
    best_params_values = results.x
    for name, val in zip(best_params_names, best_params_values):
        print(f"  {name}: {val}")

    # save results to CSV
    results_data = []
    for i in range(len(results.x_iters)):
        trial_params = {name: results.x_iters[i][j] for j, name in enumerate(best_params_names)}
        trial_params['validation_rmse'] = results.func_vals[i]
        results_data.append(trial_params)
    
    results_df = pd.DataFrame(results_data)
    try:
        results_df.to_csv(RESULTS_FILE, index=False)
        print(f"\nOptimization results saved to {RESULTS_FILE}")
        print("\nTop 5 Trials (Sorted by RMSE):")
        print(results_df.sort_values(by='validation_rmse').head())
    except IOError as e:
        print(f"Error saving results to CSV: {e}")
    except Exception as e:
        print(f"An error occurred while processing results: {e}")

if __name__ == "__main__":
    run_bayesian_optimization()