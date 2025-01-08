import torch
def noisy(x, rate=0.01):
    return x +rate * torch.randn_like(x)
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets):
        """
        Custom PyTorch dataset for time series data with corresponding targets.

        Parameters:
            data (pd.DataFrame): Input data with columns ['t', 'symbol_id', ...features].
            targets (pd.DataFrame): Target data with columns ['t', 'symbol_id', 'responder'].
        """
        self.data_dict = {t: group.drop(columns=['t', 'symbol_id']).to_numpy()
                          for t, group in data.groupby('t')}
        self.symbol_dict = {t: group['symbol_id'].values for t, group in data.groupby('t')}
        self.target_dict = {t: group['responder'].values for t, group in targets.groupby('t')}
        self.time_indices = sorted(self.data_dict.keys())

    def __len__(self):
        return len(self.time_indices)

    def __getitem__(self, index):
        t_idx = self.time_indices[index]
        data_t = self.data_dict[t_idx]
        symbols = self.symbol_dict[t_idx]
        target_t = self.target_dict[t_idx]
        return (
            torch.tensor(data_t, dtype=torch.float32),
            torch.tensor(target_t, dtype=torch.float32),
            torch.tensor(symbols, dtype=torch.long)  # Use tensors for easier batching
        )

def collate_fn(batch):
    """
    Optimized collate function to handle variable-length data efficiently.
    Pads data and targets to the maximum length in the batch using PyTorch operations.
    """
    # Extract data, targets, and symbols
    data, targets, symbols = zip(*batch)

    # Determine maximum sequence length in this batch
    max_len = max([d.size(0) for d in data])

    # Pad data and targets
    data_padded = torch.stack([
        torch.cat([d, torch.zeros(max_len - d.size(0), d.size(1))]) if d.size(0) < max_len else d
        for d in data
    ])
    targets_padded = torch.stack([
        torch.cat([t, torch.zeros(max_len - t.size(0))]) if t.size(0) < max_len else t
        for t in targets
    ])

    # Create a mask for valid positions
    mask = torch.stack([
        torch.cat([torch.ones(d.size(0)), torch.zeros(max_len - d.size(0))]) if d.size(0) < max_len else torch.ones(d.size(0))
        for d in data
    ])

    return data_padded, targets_padded, mask, symbols