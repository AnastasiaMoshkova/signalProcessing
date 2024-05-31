import pytorch_lightning as pl
import numpy as np
import os
import pandas as pd
from typing import Optional
from torch.utils.data import DataLoader
from hydra.utils import instantiate


class DataModule(pl.LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config


    def prepare_data(self):
        self.data = pd.read_csv(self.config["dataset"]["data_path"])
        self.data = self.data.drop('Unnamed: 0', axis=1)



    def setup(self, stage: Optional[str] = None) -> None:
        '''train_transfrom = instantiate(self.config.dataset.train_transform)
        val_transfrom = instantiate(self.config.dataset.val_transform)
        test_transfrom = instantiate(self.config.dataset.test_transform)'''
        #target_transform = instantiate(self.config.dataset.target_transform)

        train_dataset = self.data[self.data['split'] == 'trian'].reset_index(drop=True)
        val_dataset = self.data[self.data['split'] == 'val'].reset_index(drop=True)
        test_dataset = self.data[self.data['split'] == 'test'].reset_index(drop=True)

        self.train_data = instantiate(
            self.config.dataset.train_dataset,
            data=train_dataset,
            #transform=train_transfrom,
            #target_transform=target_transform
        )

        self.val_data = instantiate(
            self.config.dataset.val_dataset,
            data=val_dataset,
            #transform=val_transfrom,
            #target_transform=target_transform
        )

        self.test_data = instantiate(
            self.config.dataset.test_dataset,
            data=test_dataset,
            #transform=test_transfrom,
            #target_transform=target_transform
        )


    def train_dataloader(self) -> DataLoader:
        if self.config.dataset.train_dataloader._target_ is not None:
            return instantiate(
                self.config.dataset.train_dataloader,
                dataset=self.train_data,
                drop_last=True
            )

    def val_dataloader(self) -> DataLoader:
        return instantiate(
            self.config.dataset.val_dataloader,
            dataset=self.val_data,
            drop_last=True
        )

    def test_dataloader(self) -> DataLoader:
        return instantiate(
            self.config.dataset.test_dataloader,
            dataset=self.test_data,
            drop_last = True
        )

