import pandas as pd

class Dataloader:
    @staticmethod
    def _parse_stupid_format(file):
        sid_lst, pid_lst, rating_lst = [], [], []
        with open(file) as f:
            f.readline()
            for line in f.readlines():
                line = line.strip()
                sid, second = line.split('_')
                pid, rating = second.split(',')
                sid_lst.append(int(sid))
                pid_lst.append(int(pid))
                rating_lst.append(int(rating))
        return pd.DataFrame({
            'sid': sid_lst,
            'pid': pid_lst,
            'rating': rating_lst
        })

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
