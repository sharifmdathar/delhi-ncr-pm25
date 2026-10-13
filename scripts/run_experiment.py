"""Main training script for the PM2.5 forecasting project.

Runs experiments using the predefined configs and logs results.
"""

import yaml
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import load_dataset
from src.models import (
    PersistenceModel,
    HistAverageModel,
    LightGBMBaseline,
    UnivariateLSTM,
    MultivariateLSTM,
    Transformer,
    STGCNC,
    GraphWaveNet,
)
from src.models.stgcn import build_stgcn_model
from src.models.graph_wavenet import build_graph_wavenet_model


class Dataset(Dataset):
    """PyTorch Dataset for PM2.5 forecasting."""
    
    def __init__(self, dataset_path, transform=None):
        self.dataset = dataset_path
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.transform:
            item = self.transform(item)
        return item


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    """Main training loop."""
    # Configuration
    config_file = Path("configs/graph_wavenet.yaml")
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    dataset_path = Path("data/processed/delhi_ncr_pm25_2019_2024.npz")
    if dataset_path.exists():
        dataset = load_dataset(dataset_path)
        print(f"Loaded dataset with shape: {dataset}")
    else:
        print("Warning: Processed dataset not found. Using raw data.")
        # Fallback to raw data
        dataset = load_dataset(Path("data/raw/weather.csv"))
    
    # Split into train/val/test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = (
        torch.utils.data.random_split(dataset, [train_size, train_size + val_size, test_size])
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0
    )
    
    # Initialize model
    model = build_graph_wavenet_model(config)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 3e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader))
    
    # Training loop
    best_val_loss = float("inf")
    for epoch in range(1, 21):
        print(f"\nEpoch {epoch}/20")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if epoch % 5 == 0 or epoch == 19:
            checkpoint_path = f"checkpoints/stgcn_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    print("Training complete!")


if __name__ == "__main__":
    main()