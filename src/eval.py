import numpy as np
from .torch_models import SVDpp, SVDppMLP
from .dataloader import Dataloader as CSVLoader
import torch
import math

def predict_SVDpp(model: SVDpp | SVDppMLP,
                  sids: list[int] | np.ndarray,
                  pids: list[int] | np.ndarray,
                  device: torch.device,
                  batch_size: int = 1024) -> np.ndarray:
    """
    Generates predictions for given SIDs and PIDs.
    """
    # prepare implicit data
    tbr_df = CSVLoader.load_train_tbr()

    sid_to_implicit_data = {}
    implicit_groups = tbr_df.groupby('sid')['pid'].apply(list)
    if not implicit_groups.empty:
        # Calculate max length, ensure it's at least 1 for padding
        max_implicit_len = max(len(pids_list) for pids_list in implicit_groups.values)
        max_implicit_len = max(1, max_implicit_len)
    else:
        max_implicit_len = 1 # Default padding length if no implicit data

    # prepare implicit data for each sid
    for sid_val, pids_list in implicit_groups.items():
        length = len(pids_list)
        # Pad the list
        padded_pids = pids_list[:max_implicit_len] + [0] * (max_implicit_len - length) # Assuming 0 is padding index
        sid_to_implicit_data[sid_val] = {
            'pids': torch.tensor(padded_pids, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long)
        }

    model.eval()
    model.to(device)

    all_predictions = []
    num_samples = len(sids)
    num_batches = math.ceil(num_samples / batch_size)

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            # Slice the input numpy arrays or lists directly
            batch_sids_slice = sids[start_idx:end_idx]
            batch_pids_slice = pids[start_idx:end_idx]

            # Convert slices to tensors for the current batch
            batch_sids = torch.tensor(batch_sids_slice, dtype=torch.long)
            batch_pids = torch.tensor(batch_pids_slice, dtype=torch.long)

            # Collect implicit data for the batch
            batch_implicit_pids_list = []
            batch_implicit_lengths_list = []

            default_implicit_pids = torch.zeros(max_implicit_len, dtype=torch.long)
            default_implicit_length = torch.tensor(0, dtype=torch.long)

            for sid_val in batch_sids_slice:
                implicit_data = sid_to_implicit_data.get(sid_val)
                if implicit_data:
                    batch_implicit_pids_list.append(implicit_data['pids'])
                    batch_implicit_lengths_list.append(implicit_data['length'])
                else:
                    batch_implicit_pids_list.append(default_implicit_pids)
                    batch_implicit_lengths_list.append(default_implicit_length)

            batch_implicit_pids = torch.stack(batch_implicit_pids_list).to(device)
            batch_implicit_lengths = torch.stack(batch_implicit_lengths_list).to(device)
            batch_sids = batch_sids.to(device)
            batch_pids = batch_pids.to(device)

            # predict current batch
            predictions = model(batch_sids, batch_pids, batch_implicit_pids, batch_implicit_lengths)
            all_predictions.append(predictions.cpu())

    # concatenate all predictions
    return torch.cat(all_predictions).numpy()