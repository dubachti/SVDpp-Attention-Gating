import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

class SVDppDataset(Dataset):
    """
    SVD++ dataset supporting implicit feedback

    generates the following data:
    - sid: scientist id
    - pid: paper id
    - rating: explicit rating
    - implicit_pids: list of implicit paper ids for the current scientist
    - implicit_lengths: length of the implicit_pids list

    Parameters
    ----------
    explicit_df : pd.DataFrame
        DataFrame containing explicit ratings with columns ['sid', 'pid', 'rating']
    implicit_df : pd.DataFrame
        DataFrame containing implicit feedback with columns ['sid', 'pid']
    """
    def __init__(self, explicit_df: pd.DataFrame, implicit_df: pd.DataFrame=None):
        super().__init__()
        if implicit_df is None:
            implicit_df = pd.DataFrame(columns=['sid', 'pid'])
        self.unique_sids = pd.concat([explicit_df['sid'], implicit_df['sid']]).unique()
        self.unique_pids = pd.concat([explicit_df['pid'], implicit_df['pid']]).unique()

        self.num_scientists = len(self.unique_sids)
        self.num_papers = len(self.unique_pids)

        # Explicit data
        self.sid_idx = torch.tensor(explicit_df['sid'].values, dtype=torch.long)
        self.pid_idx = torch.tensor(explicit_df['pid'].values, dtype=torch.long)
        self.ratings = torch.tensor(explicit_df['rating'].values, dtype=torch.float32)

        self.global_mean = self.ratings.mean().item()


        # Implicit data
        implicit_groups = implicit_df.groupby('sid')['pid'].apply(list)
        self.implicit_map = {
            sid: torch.tensor(pids, dtype=torch.long)
            for sid, pids in implicit_groups.items()
        }
        # used for default value
        self.empty_implicit = torch.tensor([], dtype=torch.long)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        sid = self.sid_idx[idx]
        pid = self.pid_idx[idx]
        rating = self.ratings[idx]

        # Get all implicit pids for the current sid
        implicit_pids_for_s = self.implicit_map.get(sid.item(), self.empty_implicit)

        return sid, pid, rating, implicit_pids_for_s
    
def svdpp_collate_fn(batch):
    """
    Collate function required by the implicite data
    """
    # Split the batch
    sid_idx, pid_idx, ratings, implicit_pid_lists = zip(*batch)

    # Stack explicit data
    sid_batch = torch.stack(sid_idx)
    pid_batch = torch.stack(pid_idx)
    rating_batch = torch.stack(ratings)

    # Pad implicit data
    implicit_lengths = torch.tensor([len(pids) for pids in implicit_pid_lists], dtype=torch.long)
    implicit_pids_batch = pad_sequence(implicit_pid_lists, batch_first=True)

    return sid_batch, pid_batch, rating_batch, implicit_pids_batch, implicit_lengths
