import numpy as np
from .torch_models import SVDpp
from .dataloader import Dataloader as CSVLoader
import torch
import math

def predict_SVDpp(model: SVDpp,
                  sids: list[int] | np.ndarray,
                  pids: list[int] | np.ndarray,
                  device: torch.device,
                  batch_size: int = 1024) -> np.ndarray:
    """
    Generate rating predictions using the SVD++ model for given user and item IDs.
    """

    # Load implicit feedback data
    tbr_df = CSVLoader.load_train_tbr()

    # Create a mapping from SIDs to their list of implicit feedback item PIDs
    sid_to_implicit_data = {}
    implicit_groups = tbr_df.groupby('sid')['pid'].apply(list)

    # Determine the maximum length of implicit feedback lists to pad shorter lists
    if not implicit_groups.empty:
        # Calculate max length, ensure it's at least 1 for padding
        max_implicit_len = max(len(pids_list) for pids_list in implicit_groups.values)
        max_implicit_len = max(1, max_implicit_len)  # Ensure at least length 1
    else:
        max_implicit_len = 1  # Default padding length if no implicit data

    # For each user, pad their implicit item lists to the max length and store as tensors
    for sid_val, pids_list in implicit_groups.items():
        length = len(pids_list)
        padded_pids = pids_list[:max_implicit_len] + [0] * (max_implicit_len - length)
        sid_to_implicit_data[sid_val] = {
            'pids': torch.tensor(padded_pids, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long)
        }

    # Prepare the model for evaluation and move it to the specified device
    model.eval()
    model.to(device)

    all_predictions = []
    num_samples = len(sids)
    num_batches = math.ceil(num_samples / batch_size)

    # Disable gradient calculation since this is inference only
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            # Slice batch inputs and convert to tensors
            batch_sids_slice = sids[start_idx:end_idx]
            batch_pids_slice = pids[start_idx:end_idx]

            batch_sids = torch.tensor(batch_sids_slice, dtype=torch.long)
            batch_pids = torch.tensor(batch_pids_slice, dtype=torch.long)

            # Prepare batches of implicit feedback tensors, padding missing users with zeros
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

            # Stack lists into batch tensors and move to device
            batch_implicit_pids = torch.stack(batch_implicit_pids_list).to(device)
            batch_implicit_lengths = torch.stack(batch_implicit_lengths_list).to(device)
            batch_sids = batch_sids.to(device)
            batch_pids = batch_pids.to(device)

            # Get model predictions for the batch
            predictions = model(batch_sids, batch_pids, batch_implicit_pids, batch_implicit_lengths)

            # Collect predictions on CPU
            all_predictions.append(predictions.cpu())

    # Concatenate all batch predictions and convert to numpy array
    return torch.cat(all_predictions).numpy()
