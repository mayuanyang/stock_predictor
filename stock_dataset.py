import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, data, sequence_length, target_length):
        """
        Args:
            data: Normalized stock data including cyclical date features.
            sequence_length: The number of past days to look back for the input sequence.
            target_length: The number of future days to predict.
        """
        self.data = data
        self.sequence_length = sequence_length
        self.target_length = target_length
        self.sequences, self.targets = self._create_sequences()

    def _create_sequences(self):
        sequences = []
        targets = []
        for i in range(len(self.data) - self.sequence_length - self.target_length):
            sequences.append(self.data[i:i + self.sequence_length])
            targets.append(self.data[i + self.sequence_length:i + self.sequence_length + self.target_length, -1])  # Predict 'Adj Close'
        return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]
