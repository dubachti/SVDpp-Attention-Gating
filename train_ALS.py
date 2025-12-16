"""Training script for the Alternating Least Squares (ALS) model.

Train the model, evaluate on test set, and generate predictions for competition submission.
"""

import torch
from sklearn.model_selection import train_test_split
from src.dataloader import Dataloader
from src.models.ALS import *
from src.config import load_als_config


def main():
    # Load configuration
    config = load_als_config()
    
    # Model hyperparameters
    REG_LAMBDA = config['model']['reg_lambda']
    EMBEDDING_DIM = config['model']['embedding_dim']
    N_ITERATIONS = config['model']['n_iterations']
    
    # Data split parameters
    TEST_SIZE = config['data']['test_size']
    VALIDATION_SIZE = config['data']['validation_size']
    RANDOM_STATE = config['data']['random_state']
    
    # Preprocessing
    NORMALIZE_ROWS = config['preprocessing']['normalize_rows']
    NORMALIZE_COLUMNS = config['preprocessing']['normalize_columns']
    
    # Output
    SUBMISSION_FILE = config['output']['submission_file']
    
    # Load data
    print(" > READING DATA...")
    ratings_df = Dataloader.load_train_ratings()
    ratings_temp, ratings_test = train_test_split(ratings_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)    
    ratings_train, _ = train_test_split(ratings_temp, test_size=VALIDATION_SIZE/(1-TEST_SIZE), random_state=RANDOM_STATE)
    
    init_train_mat = torch.tensor(ratings_train.pivot(index="sid", columns="pid", values="rating").values, dtype=torch.float32)
    print(" > DATA READ")

    if NORMALIZE_ROWS:
        train_mat, mean, std = center_and_normalize_rows(init_train_mat)
        print(" > Normalized and centered each column")
    elif NORMALIZE_COLUMNS:
        train_mat, mean, std = center_and_normalize_columns(init_train_mat)
        print(" > Normalized and centered each row")
    else:
        train_mat = init_train_mat

    model = ALS(
        device="cuda" if torch.cuda.is_available() else "cpu",
        train_mat=train_mat,
    )

    print(" > Training model...")
                
    model.train(lam=REG_LAMBDA, rank=EMBEDDING_DIM, iterations=N_ITERATIONS)
    predictions_matrix = model.get_predictions_matrix()

    if NORMALIZE_ROWS:
        predictions_matrix = recover_matrix_rows(predictions_matrix, mean, std)
    elif NORMALIZE_COLUMNS:
        predictions_matrix = recover_matrix_columns(predictions_matrix, mean, std)
    else:
        raise ValueError("Normalization must be enabled for either rows or columns.")

    loss = evaluate_prediction_matrix(ratings_test, predictions_matrix)
    print(f"Test RMSE: {loss}")

    # Create submission
    print(" > CREATING SUBMISSION...")
    pred_fn = lambda sids, pids: predictions_matrix[sids, pids]
    make_submission(pred_fn, SUBMISSION_FILE)

    print(" > END")

if __name__ == "__main__":
    main()

