import numpy as np
import re
import torch
from torch.utils.data import Dataset
from typing import Optional, Any


class SignalDataset(Dataset):
    def __init__(
            self,
            data,
            # transform: Optional[Any] = None,
            # target_transform: Optional[Any] = None
    ) -> None:
        self.signal_data = data
        self.cache = dict()

        # self.transform = transform
        # self.target_transform = target_transform

    def _preprocessing_signal(self, sample):
        signal = sample[0:1600]
        target = sample['target']
        number = sample['number']

        return signal, target, number

    def _get_item(self, idx):
        if idx in self.cache.keys():
            return self.cache[idx]
        # print(type(self.signal_data), len(self.signal_data))
        sample = self.signal_data.loc[idx]
        sample = self._preprocessing_signal(sample)
        self.cache[idx] = sample
        return sample

    def __getitem__(self, idx):
        signal, target, number = self._get_item(idx)
        return torch.tensor(signal).to(torch.float32), torch.tensor(target).to(torch.int), torch.tensor(number).to(torch.int)

    def __len__(self):
        return len(self.signal_data)

