#!/usr/bin/env python3
"""
Air-Gapped NOC Copilot - LSTM Model for Predictive Fault Detection
Time-series forecasting model using LSTM networks for network failure prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List, Optional
import json
from pathlib import Path
from datetime import datetime


# ════════════════════════════════════════════════════════════════
# LSTM Model Architecture
# ════════════════════════════════════════════════════════════════

class LSTMPredictor(nn.Module):
    """LSTM-based predictor for network fault forecasting."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        forecast_horizon: int = 10,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.forecast_horizon = forecast_horizon
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Attention mechanism
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(lstm_output_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size * forecast_horizon)
        
        # Regularization
        self.dropout_layer = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        
        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
            Predictions of shape (batch_size, forecast_horizon, output_size)
        """
        batch_size = x.size(0)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        lstm_out = self.attention_norm(lstm_out + attn_out)
        
        # Take the last time step
        out = lstm_out[:, -1, :]
        
        # Fully connected layers with residual connections
        out = self.dropout_layer(out)
        out = self.fc1(out)
        out = self.batch_norm1(out)
        out = self.relu(out)
        
        out = self.dropout_layer(out)
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        
        # Output projection
        out = self.fc3(out)
        out = out.view(batch_size, self.forecast_horizon, self.output_size)
        
        return out


# ════════════════════════════════════════════════════════════════
# Multi-Task LSTM for Anomaly Detection
# ════════════════════════════════════════════════════════════════

class LSTMMultiTask(nn.Module):
    """Multi-task LSTM for prediction and anomaly detection."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        forecast_horizon: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # Shared LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Task-specific heads
        # Task 1: Utilization forecasting
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon),
        )
        
        # Task 2: Anomaly detection
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon),
            nn.Sigmoid(),
        )
        
        # Task 3: Time-to-impact estimation
        self.tti_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon),
            nn.Softplus(),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multiple task heads.
        
        Returns:
            Dictionary with 'forecast', 'anomaly_prob', 'tti_estimates'
        """
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        
        return {
            "forecast": self.forecast_head(out).unsqueeze(-1),
            "anomaly_prob": self.anomaly_head(out),
            "tti_estimates": self.tti_head(out),
        }


# ════════════════════════════════════════════════════════════════
# Training Loop
# ════════════════════════════════════════════════════════════════

class LSTMTrainer:
    """Trains the LSTM model for network fault prediction."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
        }
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss for prediction."""
        # Align trailing dims: models output (batch, H, 1) but targets are (batch, H)
        if predictions.dim() == targets.dim() + 1 and predictions.size(-1) == 1:
            predictions = predictions.squeeze(-1)

        # MSE loss
        mse = self.mse_loss(predictions, targets)

        # MAE loss
        mae = self.mae_loss(predictions, targets)

        # Combined loss
        loss = 0.7 * mse + 0.3 * mae

        return loss
    
    def train_epoch(
        self, train_loader: DataLoader
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_X)
            
            # Compute loss
            loss = self.compute_loss(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics (squeeze to match targets shape)
            pred_sq = predictions.squeeze(-1) if predictions.dim() > batch_y.dim() else predictions
            total_loss += loss.item()
            total_mae += self.mae_loss(pred_sq, batch_y).item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            predictions = self.model(batch_X)
            loss = self.compute_loss(predictions, batch_y)
            
            pred_sq = predictions.squeeze(-1) if predictions.dim() > batch_y.dim() else predictions
            total_loss += loss.item()
            total_mae += self.mae_loss(pred_sq, batch_y).item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        patience: int = 15,
    ) -> Dict:
        """Complete training loop with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n{'='*60}")
        print(f"LSTM Training - {num_epochs} epochs, patience={patience}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_mae = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_mae = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_mae"].append(train_mae)
            self.history["val_mae"].append(val_mae)
            
            # Print progress
            lr = self.optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | "
                f"LR: {lr:.2e}"
            )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint("best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.load_checkpoint("best_model.pth")
        
        print(f"\n{'='*60}")
        print(f"Training Complete! Best Val Loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
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
# Model Inference
# ════════════════════════════════════════════════════════════════

class LSTMInferenceEngine:
    """Inference engine for trained LSTM model."""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def predict(
        self, x: torch.Tensor
    ) -> np.ndarray:
        """Make predictions on input sequence."""
        x = x.to(self.device)
        predictions = self.model(x)
        return predictions.cpu().numpy()
    
    def predict_single(
        self, sequence: np.ndarray
    ) -> Dict[str, any]:
        """Predict for a single input sequence."""
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        prediction = self.predict(x)[0]
        
        return {
            "forecast": prediction,
            "max_predicted": float(np.max(prediction)),
            "mean_predicted": float(np.mean(prediction)),
            "min_predicted": float(np.min(prediction)),
        }
    
    def detect_anomaly(
        self,
        sequence: np.ndarray,
        threshold: float = 0.8,
    ) -> Dict[str, any]:
        """Detect anomalies in the input sequence."""
        result = self.predict_single(sequence)
        
        # Simple threshold-based anomaly detection
        anomaly_score = float(result["max_predicted"] > threshold)
        
        return {
            "is_anomaly": bool(anomaly_score),
            "anomaly_score": anomaly_score,
            "max_prediction": result["max_predicted"],
            "threshold": threshold,
        }


# ════════════════════════════════════════════════════════════════
# Utility Functions
# ════════════════════════════════════════════════════════════════

def create_model_config(
    input_size: int,
    forecast_horizon: int = 10,
) -> Dict:
    """Create default model configuration."""
    return {
        "input_size": input_size,
        "hidden_size": 128,
        "num_layers": 2,
        "output_size": 1,
        "dropout": 0.2,
        "bidirectional": False,
        "forecast_horizon": forecast_horizon,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "num_epochs": 100,
        "patience": 15,
    }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ════════════════════════════════════════════════════════════════
# Main Entry Point
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example usage
    config = create_model_config(input_size=25, forecast_horizon=10)
    
    model = LSTMPredictor(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        output_size=config["output_size"],
        dropout=config["dropout"],
        bidirectional=config["bidirectional"],
        forecast_horizon=config["forecast_horizon"],
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Config: {json.dumps(config, indent=2)}")
