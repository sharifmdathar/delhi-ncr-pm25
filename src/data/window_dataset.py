"""Window Dataset for on-the-fly window slicing from continuous arrays."""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional


class WindowDS(Dataset):
    """Slice (input, target, mask) windows from continuous arrays on-the-fly.
    
    Args:
        X_full: [T, N, F] input features (normalized)
        y_full: [T, N] target (normalized)
        mask_X: [T, N, F] input masks
        mask_y: [T, N] target masks
        start_idx: first valid window start index
        end_idx: first invalid window start index (exclusive)
        input_hours: length of input window
        output_hours: length of output window
    """
    
    def __init__(
        self,
        X_full: np.ndarray,
        y_full: np.ndarray,
        mask_X: np.ndarray,
        mask_y: np.ndarray,
        start_idx: int,
        end_idx: int,
        input_hours: int = 72,
        output_hours: int = 48,
    ):
        self.X_full = X_full
        self.y_full = y_full
        self.mask_X = mask_X
        self.mask_y = mask_y
        self.input_hours = input_hours
        self.output_hours = output_hours
        
        # Valid window starts: [start_idx, end_idx - input_hours - output_hours]
        max_start = end_idx - input_hours - output_hours
        if max_start <= start_idx:
            raise ValueError(f"Split too small: need at least {input_hours + output_hours} hours")
        
        self.window_starts = np.arange(start_idx, max_start + 1, dtype=np.int64)
    
    def __len__(self) -> int:
        return len(self.window_starts)
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.window_starts[i]
        m = s + self.input_hours
        e = m + self.output_hours
        
        # X: [input_hours, N, F] - use torch.tensor instead of from_numpy for numpy 2.x compatibility
        x = torch.tensor(self.X_full[s:m], dtype=torch.float32)
        mask_x = torch.tensor(self.mask_X[s:m], dtype=torch.float32)
        
        # y: [output_hours, N]
        y = torch.tensor(self.y_full[m:e], dtype=torch.float32)
        mask_y = torch.tensor(self.mask_y[m:e], dtype=torch.float32)
        
        return x, y, mask_x, mask_y


def create_split_datasets(
    dataset: Dict,
    split: str,
    input_hours: int = 72,
    output_hours: int = 48,
) -> WindowDS:
    """Create WindowDS for a given split."""
    key_map = {"train": "split_indices_train", "val": "split_indices_val", "test": "split_indices_test"}
    key = key_map.get(split)
    if key is None or key not in dataset:
        raise ValueError(f"Unknown split: {split}. Available keys: {[k for k in dataset.keys() if 'split' in k]}")
    
    idx = dataset[key]
    start_idx = int(idx[0])
    end_idx = int(idx[-1]) + 1
    
    return WindowDS(
        X_full=dataset["X_full"],
        y_full=dataset["y_full"],
        mask_X=dataset["mask_X"],
        mask_y=dataset["mask_y"],
        start_idx=start_idx,
        end_idx=end_idx,
        input_hours=input_hours,
        output_hours=output_hours,
    )


def get_all_splits(
    dataset: Dict,
    input_hours: int = 72,
    output_hours: int = 48,
) -> Tuple[WindowDS, WindowDS, WindowDS]:
    """Create train/val/test WindowDS."""
    train_ds = create_split_datasets(dataset, "train", input_hours, output_hours)
    val_ds = create_split_datasets(dataset, "val", input_hours, output_hours)
    test_ds = create_split_datasets(dataset, "test", input_hours, output_hours)
    return train_ds, val_ds, test_ds


# Quick test
if __name__ == "__main__":
    from src.data.dataset import load_dataset
    ds = load_dataset("data/processed/delhi_ncr_pm25_2019_2024.npz")
    train, val, test = get_all_splits(ds)
    print(f"Train windows: {len(train)}")
    print(f"Val windows: {len(val)}")
    print(f"Test windows: {len(test)}")
    x, y, mx, my = train[0]
    print(f"Sample shapes: x={x.shape}, y={y.shape}, mx={mx.shape}, my={my.shape}")