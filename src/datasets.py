from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ASLDataset(Dataset):
    def __init__(self, data_dir):
        data_dir = Path(data_dir)

        self.sequences = np.load(data_dir / "sequences.npy", mmap_mode="r")
        self.labels = np.load(data_dir / "labels.npy", mmap_mode="r")

        if len(self.sequences) != len(self.labels):
            raise ValueError(
                f"Sequence length {len(self.sequences)} and labels length {len(self.labels)} are no equal."
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sequence = torch.from_numpy(self.sequences[index].copy()).float()

        sequence = np.nan_to_num(
            sequence,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        label = torch.tensor(
            int(self.labels[index]),
            dtype=torch.long,
        )

        return sequence, label
