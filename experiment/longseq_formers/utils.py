import logging
import sys

import pytorch_lightning as pl
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .dataset.language_modeling import LanguageModelingDataset


def get_logger(name: str) -> logging.Logger:
    """Return logger for logging

    Args:
        name: logger name
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(handler)
    return logger


class BatchedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: LanguageModelingDataset,
        valid_dataset: LanguageModelingDataset,
        shuffle: bool,
        distributed: bool = True,
    ) -> None:
        super().__init__()

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.shuffle = shuffle
        self.distributed = distributed

    def train_dataloader(self):
        # Use batch size as 1 because already batched
        if self.distributed:
            sampler = DistributedSampler(self.train_dataset, shuffle=self.shuffle)
        elif self.shuffle:
            sampler = RandomSampler(self.train_dataset)
        else:
            sampler = SequentialSampler(self.train_dataset)
        return DataLoader(self.train_dataset, batch_size=1, sampler=sampler, collate_fn=self.train_dataset.collate_fn)

    def val_dataloader(self):
        if self.distributed:
            sampler = DistributedSampler(self.valid_dataset, shuffle=False)
        else:
            sampler = SequentialSampler(self.valid_dataset)
        return DataLoader(self.valid_dataset, batch_size=1, sampler=sampler, collate_fn=self.valid_dataset.collate_fn)
