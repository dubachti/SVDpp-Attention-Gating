"""Training script for SVDppAG model.

Train the model, eval on test set, and generate predictions for competition submission.
"""
from src.SVDpp_dataset import SVDppDataset, svdpp_collate_fn
from src.models.SVDppAG import SVDppAG
from src.SVDpp_trainer import SVDppTrainer
from src.dataloader import Dataloader as CSVLoader
from src.eval import predict_SVDpp
from src.config import load_svdppag_config
from datetime import datetime
from torch import nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split as sklearn_split


def main():
    # Load configuration
    config = load_svdppag_config()
    
    # Model hyperparameters
    EMBEDDING_DIM = config['model']['embedding_dim']
    
    # Training hyperparameters
    BATCH_SIZE = config['training']['batch_size']
    LEARNING_RATE = config['training']['learning_rate']
    REG_LAMBDA = config['training']['reg_lambda']
    NUM_EPOCHS = config['training']['num_epochs']
    NUM_WORKERS = config['training']['num_workers']
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = config['early_stopping']['patience']
    MIN_DELTA_IMPROVEMENT = config['early_stopping']['min_delta']
    
    # Scheduler
    SCHEDULER_FACTOR = config['scheduler']['factor']
    SCHEDULER_PATIENCE = config['scheduler']['patience']
    SCHEDULER_MIN_LR = config['scheduler']['min_lr']
    
    # Data split parameters
    TEST_SIZE = config['data']['test_size']
    VALIDATION_SIZE = config['data']['validation_size']
    RANDOM_STATE = config['data']['random_state']
    
    # Output
    MODEL_FILE = config['output']['model_file']
    # 1. load data
    print(" > READING DATA...")
    ratings_df = CSVLoader.load_train_ratings()
    tbr_df = CSVLoader.load_train_tbr()

    ratings_train, ratings_test = sklearn_split(ratings_df, test_size=0.25, random_state=42)
    print(" > DATA READ")

    # 2. build datasets and dataloaders
    print(" > LOADING DATA...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ratings_temp, ratings_test = sklearn_split(ratings_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)    
    ratings_train, ratings_valid = sklearn_split(ratings_temp, test_size=VALIDATION_SIZE/(1-TEST_SIZE), random_state=RANDOM_STATE)
    
    print(f"Training data size: {len(ratings_train)}, Validation data size: {len(ratings_valid)}, Test data size: {len(ratings_test)}")

    train_dataset = SVDppDataset(ratings_train, tbr_df)
    valid_dataset = SVDppDataset(ratings_valid, tbr_df)
    test_dataset = SVDppDataset(ratings_test, tbr_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=svdpp_collate_fn, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE*2, shuffle=False, collate_fn=svdpp_collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, collate_fn=svdpp_collate_fn, num_workers=NUM_WORKERS)
    print(" > DATA LOADED")

    model = SVDppAG(
        num_scientists=train_dataset.num_scientists,
        num_papers=train_dataset.num_papers,
        embedding_dim=EMBEDDING_DIM,
        global_mean=train_dataset.global_mean
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss(reduction='sum')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)

    # 4. train
    trainer = SVDppTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        reg_lambda=REG_LAMBDA,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        scheduler=scheduler,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        min_delta=MIN_DELTA_IMPROVEMENT,
    )

    trainer.train(num_epochs=NUM_EPOCHS)

    print("Evaluating on the final test set...")
    test_rmse = trainer.evaluate_on_loader(test_loader)
    print(f"Test RMSE: {test_rmse:.4f}")

    trainer.save_model(MODEL_FILE)

    # 5. predict
    print("Predicting submission data...")
    submission_df = CSVLoader.load_sample_submission()

    out = predict_SVDpp(model=model,
                        sids=submission_df['sid'].values,
                        pids=submission_df['pid'].values,
                        device=device)
    submission_df['rating'] = out
    submission_df['rating'] = submission_df['rating'].clip(1, 5)

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'{datetime_str}_submission.csv', 'w') as f:
        f.write('sid_pid,rating\n')
        for _, row in submission_df.iterrows():
            f.write(f'{int(row["sid"])}_{int(row["pid"])},{row["rating"]}\n')

    print(f"Submission file created: {datetime_str}_submission.csv")

if __name__ == "__main__":
    main()