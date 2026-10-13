"""Training utilities for PM2.5 forecasting models."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional


class Trainer:
    """Handles training and validation of forecasting models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()  # Default; override if needed
        
    def set_model(self, model: torch.nn.Module):
        """Set the model to train."""
        self.model = model.to(self.device)
        
    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """Set the optimizer."""
        self.optimizer = optimizer
        
    def set_scheduler(self, scheduler):
        """Set the learning rate scheduler."""
        self.scheduler = scheduler
        
    def train_one_epoch(
        self, 
        dataloader: torch.utils.data.DataLoader, 
        epoch: int,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    @torch.no_grad()
    def validate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Validate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
    ) -> Dict[str, Any]:
        """Full training loop with checkpointing."""
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    self.config.get("checkpoint_path", "checkpoints/best.pt"),
                )
                
        return history
    
    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run inference on given inputs."""
        self.model.eval()
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        return outputs.cpu()


class EarlyStopper:
    """Early stopping based on validation loss."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        
    def step(self, val_loss: float) -> bool:
        """Step the early stopping counter.
        
        Returns True if training should stop.
        """
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience