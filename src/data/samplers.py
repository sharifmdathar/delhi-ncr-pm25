"""Custom samplers that work without numpy (for torch 2.1 + numpy 2.x compatibility)."""

import torch
from torch.utils.data import Sampler
from typing import Iterator, Optional, Sequence, List
import math


class NoNumpyRandomSampler(Sampler[int]):
    """Random sampler that doesn't require numpy (avoids torch.randperm().numpy() issue)."""
    
    def __init__(self, data_source: int, generator: Optional[torch.Generator] = None):
        self.data_source = data_source
        self.generator = generator
    
    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source) if hasattr(self.data_source, '__len__') else self.data_source
        # Use torch.randperm without .numpy() - yield directly from tensor
        perm = torch.randperm(n, generator=self.generator)
        for idx in perm:
            yield int(idx)
    
    def __len__(self) -> int:
        return len(self.data_source) if hasattr(self.data_source, '__len__') else self.data_source


def create_dataloaders(train_ds, val_ds, batch_size=32, shuffle_train=True):
    """Create DataLoaders with numpy-compatible samplers."""
    
    if shuffle_train:
        train_sampler = NoNumpyRandomSampler(len(train_ds))
    else:
        train_sampler = torch.utils.data.SequentialSampler(train_ds)
    
    val_sampler = torch.utils.data.SequentialSampler(val_ds)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, sampler=val_sampler, num_workers=0
    )
    
    return train_loader, val_loader