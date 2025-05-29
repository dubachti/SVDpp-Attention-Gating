import pandas as pd

class Dataloader:
    """
    Utility class providing static methods for loading, processing, and writing data files.
    """
    @staticmethod
    def _parse_stupid_format(file):
        df = pd.read_csv(file)
        df[["sid", "pid"]] = df["sid_pid"].str.split("_", expand=True)
        df = df.drop("sid_pid", axis=1)
        df["sid"] = df["sid"].astype(int)
        df["pid"] = df["pid"].astype(int)
        return df

    @staticmethod
    def load_train_ratings(path="data/train_ratings.csv"):
        return Dataloader._parse_stupid_format(path)

    @staticmethod
    def load_train_tbr(path="data/train_tbr.csv"):
        df = pd.read_csv(path)
        return df.astype({'sid': int, 'pid': int})
    
    @staticmethod
    def load_sample_submission(path="data/sample_submission.csv"):
        return Dataloader._parse_stupid_format(path)
    
    @staticmethod
    def make_submission(pred_fn, sample_submission_path = "data/sample_submission.csv"):
        df = Dataloader.load_sample_submission(sample_submission_path)
        sids = df["sid"].values
        pids = df["pid"].values
        df["rating"] = pred_fn(sids, pids)
        df.to_csv(sample_submission_path, index=False)
