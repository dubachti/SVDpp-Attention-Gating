import numpy as np
from .torch_models import SVDpp, SVDppMLP
from .dataloader import Dataloader as CSVLoader
import torch

def predict_SVDpp(model: SVDpp | SVDppMLP, 
                  sids: list[int], 
                  pids: list[int],
                  device: torch.device) -> np.ndarray:
    """
    Generates predictions for given SIDs and PIDs.
    """
    # prepare implicit data
    tbr_df = CSVLoader.load_train_tbr()

    sid_to_implicit_data = {}
    implicit_groups = tbr_df.groupby('sid')['pid'].apply(list)
    max_implicit_len = max(len(pids) for pids in implicit_groups.values)
    for sid, pids in implicit_groups.items():
        length = len(pids)
        # Pad the list
        padded_pids = pids[:max_implicit_len] + [0] * (max_implicit_len - length) # Assuming 0 is padding index
        sid_to_implicit_data[sid] = {
            'pids': torch.tensor(padded_pids, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long)
        }

    model.eval()
    model.to(device)

    batch_sids = torch.tensor(sids, dtype=torch.long)
    batch_pids = torch.tensor(pids, dtype=torch.long)
    
    # Collect implicit data for the batch
    batch_implicit_pids = []
    batch_implicit_lengths = []
    max_len_in_map = 0 # Determine max length from the map if needed, or use the training max_len
    if sid_to_implicit_data:
         # Get the length of the first item's pids tensor to know the padding length
         first_key = next(iter(sid_to_implicit_data))
         max_len_in_map = sid_to_implicit_data[first_key]['pids'].shape[0]


    for sid in sids:
        if sid in sid_to_implicit_data:
            implicit_data = sid_to_implicit_data[sid]
            batch_implicit_pids.append(implicit_data['pids'])
            batch_implicit_lengths.append(implicit_data['length'])
        else:
            batch_implicit_pids.append(torch.zeros(max_len_in_map, dtype=torch.long)) 
            batch_implicit_lengths.append(torch.tensor(0, dtype=torch.long))

    batch_implicit_pids = torch.stack(batch_implicit_pids).to(device)
    batch_implicit_lengths = torch.stack(batch_implicit_lengths).to(device)
    batch_sids = batch_sids.to(device)
    batch_pids = batch_pids.to(device)

    with torch.no_grad():
        predictions = model(batch_sids, batch_pids, batch_implicit_pids, batch_implicit_lengths)
        
    return predictions.cpu().numpy()