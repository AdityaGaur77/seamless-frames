#!/usr/bin/env python3
"""
Air-Gapped NOC Copilot - TCN (Temporal Convolutional Network) Model
Time-series forecasting using dilated causal convolutions for network fault prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Optional


# ════════════════════════════════════════════════════════════════
# TCN Building Blocks
# ════════════════════════════════════════════════════════════════

class TemporalBlock(nn.Module):
    """Temporal block with dilated causal convolution."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        
        # Residual connection
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Trim the last elements of the sequence."""
    
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


# ════════════════════════════════════════════════════════════════
# TCN Model
# ════════════════════════════════════════════════════════════════

class TCNPredictor(nn.Module):
    """TCN-based predictor for network fault forecasting."""
    
    def __init__(
        self,
        input_size: int,
        num_channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        output_size: int = 1,
        forecast_horizon: int = 10,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.forecast_horizon = forecast_horizon
        
        if num_channels is None:
            num_channels = [64, 128, 128, 64]
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            
            layers.append(
                TemporalBlock(
                    in_ch, out_ch, kernel_size,
                    stride=1, dilation=dilation,
                    padding=padding, dropout=dropout,
                )
            )
        
        self.network = nn.Sequential(*layers)
        
        # Output projection
        self.fc1 = nn.Linear(num_channels[-1], num_channels[-1] // 2)
        self.fc2 = nn.Linear(num_channels[-1] // 2, output_size * forecast_horizon)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN.
        
        Args:
            x: Input tensor (batch, seq_len, features)
        
        Returns:
            Predictions (batch, forecast_horizon, output_size)
        """
        # TCN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        
        # TCN encoding
        out = self.network(x)
        
        # Take the last time step
        out = out[:, :, -1]
        
        # Output projection
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        
        # Reshape to (batch, forecast_horizon, output_size)
        batch_size = x.size(0)
        out = out.view(batch_size, self.forecast_horizon, -1)
        
        return out


# ════════════════════════════════════════════════════════════════
# Hybrid TCN-LSTM Model
# ════════════════════════════════════════════════════════════════

class TCNLSTMHybrid(nn.Module):
    """Hybrid TCN-LSTM model combining local and global patterns."""
    
    def __init__(
        self,
        input_size: int,
        tcn_channels: List[int] = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
        forecast_horizon: int = 10,
    ):
        super().__init__()
        
        self.forecast_horizon = forecast_horizon
        
        # TCN for local pattern extraction
        if tcn_channels is None:
            tcn_channels = [64, 64, 128]
        
        tcn_layers = []
        for i, out_ch in enumerate(tcn_channels):
            in_ch = input_size if i == 0 else tcn_channels[i - 1]
            dilation = 2 ** i
            padding = (3 - 1) * dilation
            tcn_layers.append(
                TemporalBlock(in_ch, out_ch, 3, 1, dilation, padding, dropout)
            )
        
        self.tcn = nn.Sequential(*tcn_layers)
        
        # LSTM for global sequence modeling
        self.lstm = nn.LSTM(
            input_size=tcn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.Tanh(),
            nn.Linear(lstm_hidden // 2, 1),
        )
        
        # Output layers
        self.fc1 = nn.Linear(lstm_hidden, lstm_hidden // 2)
        self.fc2 = nn.Linear(lstm_hidden // 2, output_size * forecast_horizon)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TCN encoding
        tcn_out = self.tcn(x.permute(0, 2, 1))
        tcn_out = tcn_out.permute(0, 2, 1)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(tcn_out)
        
        # Attention-weighted pooling
        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Output projection
        out = self.dropout(context)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out.view(-1, self.forecast_horizon, 1)


# ════════════════════════════════════════════════════════════════
# TCN Training Loop
# ════════════════════════════════════════════════════════════════

class TCNTrainer:
    """Trains the TCN model for network fault prediction."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Optimizer with layer-wise learning rate decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 5e-4),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.get("learning_rate", 5e-4),
            steps_per_epoch=config.get("steps_per_epoch", 100),
            epochs=config.get("num_epochs", 100),
            pct_start=0.1,
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.SmoothL1Loss()
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
        }
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss."""
        # Align trailing dims: models output (batch, H, 1) but targets are (batch, H)
        if predictions.dim() == targets.dim() + 1 and predictions.size(-1) == 1:
            predictions = predictions.squeeze(-1)

        # Huber loss (robust to outliers)
        huber = self.huber_loss(predictions, targets)

        # MSE loss
        mse = self.mse_loss(predictions, targets)

        return 0.5 * huber + 0.5 * mse
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(batch_X)
            
            loss = self.compute_loss(predictions, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            predictions = self.model(batch_X)
            loss = self.compute_loss(predictions, batch_y)
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        patience: int = 15,
    ) -> Dict:
        """Complete training loop."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n{'='*60}")
        print(f"TCN Training - {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            lr = self.optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"LR: {lr:.2e}"
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint("best_tcn_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        self.load_checkpoint("best_tcn_model.pth")
        
        print(f"\nTraining Complete! Best Val Loss: {best_val_loss:.4f}\n")
        return self.history
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)


# ════════════════════════════════════════════════════════════════
# TCN Inference
# ════════════════════════════════════════════════════════════════

class TCNPredictorEngine:
    """Inference engine for trained TCN model."""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Make predictions."""
        x = x.to(self.device)
        predictions = self.model(x)
        return predictions.cpu().numpy()
    
    def predict_with_confidence(
        self,
        sequence: np.ndarray,
        n_forward: int = 10,
    ) -> Dict[str, any]:
        """Predict with confidence intervals using Monte Carlo dropout."""
        self.model.train()  # Enable dropout
        
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        predictions = []
        for _ in range(n_forward):
            pred = self.predict(x)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        self.model.eval()
        
        return {
            "mean": mean_pred,
            "std": std_pred,
            "lower_bound": mean_pred - 1.96 * std_pred,
            "upper_bound": mean_pred + 1.96 * std_pred,
            "confidence_level": 0.95,
        }


# ════════════════════════════════════════════════════════════════
# Utility Functions
# ════════════════════════════════════════════════════════════════

def create_tcn_config(
    input_size: int,
    forecast_horizon: int = 10,
) -> Dict:
    """Create default TCN configuration."""
    return {
        "input_size": input_size,
        "num_channels": [64, 128, 128, 64],
        "kernel_size": 3,
        "dropout": 0.2,
        "output_size": 1,
        "forecast_horizon": forecast_horizon,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "num_epochs": 100,
        "patience": 15,
    }


def create_hybrid_config(
    input_size: int,
    forecast_horizon: int = 10,
) -> Dict:
    """Create hybrid TCN-LSTM configuration."""
    return {
        "input_size": input_size,
        "tcn_channels": [64, 64, 128],
        "lstm_hidden": 128,
        "lstm_layers": 2,
        "dropout": 0.2,
        "output_size": 1,
        "forecast_horizon": forecast_horizon,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "num_epochs": 100,
        "patience": 15,
    }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import json
    
    # Example configurations
    tcn_config = create_tcn_config(input_size=25, forecast_horizon=10)
    hybrid_config = create_hybrid_config(input_size=25, forecast_horizon=10)
    
    print("TCN Config:")
    print(json.dumps(tcn_config, indent=2))
    
    print("\nHybrid TCN-LSTM Config:")
    print(json.dumps(hybrid_config, indent=2))
    
    # Create models
    tcn_model = TCNPredictor(
        input_size=25,
        num_channels=tcn_config["num_channels"],
        kernel_size=tcn_config["kernel_size"],
        dropout=tcn_config["dropout"],
        forecast_horizon=10,
    )
    
    hybrid_model = TCNLSTMHybrid(
        input_size=25,
        tcn_channels=hybrid_config["tcn_channels"],
        lstm_hidden=hybrid_config["lstm_hidden"],
        lstm_layers=hybrid_config["lstm_layers"],
        dropout=hybrid_config["dropout"],
        forecast_horizon=10,
    )
    
    print(f"\nTCN Parameters: {count_parameters(tcn_model):,}")
    print(f"Hybrid Parameters: {count_parameters(hybrid_model):,}")
