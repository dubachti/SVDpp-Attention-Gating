from src.torch_dataset import SVDppDataset, svdpp_collate_fn
from src.torch_models import SVDpp, SVDppMLP
from src.torch_trainer import SVDppTrainer
from src.dataloader import Dataloader as CSVLoader
from src.eval import predict_SVDpp
from datetime import datetime
from torch import nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split as sklearn_split

def main():
    # 1. load data
    ratings_df = CSVLoader.load_train_ratings()
    tbr_df = CSVLoader.load_train_tbr()

    ratings_train, ratings_test = sklearn_split(ratings_df, test_size=0.25, random_state=42)


    # 2. build datasets and dataloaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 1024

    train_dataset = SVDppDataset(ratings_train, tbr_df)
    valid_dataset = SVDppDataset(ratings_test, tbr_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=svdpp_collate_fn, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE*2, shuffle=False, collate_fn=svdpp_collate_fn, num_workers=2)

    # 3. hyperparams
    EMBEDDING_DIM = 30
    LEARNING_RATE = 0.003
    REG_LAMBDA = 0.01
    NUM_EPOCHS = 8

    model = SVDppMLP(
        num_scientists=train_dataset.num_scientists,
        num_papers=train_dataset.num_papers,
        embedding_dim=EMBEDDING_DIM,
        global_mean=train_dataset.global_mean
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss(reduction='sum')

    # 4. train
    trainer = SVDppTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        reg_lambda=REG_LAMBDA,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )

    trainer.train(num_epochs=NUM_EPOCHS)

    trainer.save_model("svdpp_mlp_model.pth")

    # 5. predict
    print("Predicting...")
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
        for i, row in submission_df.iterrows():
            f.write(f'{int(row["sid"])}_{int(row["pid"])},{row["rating"]}\n')


if __name__ == "__main__":
    main()