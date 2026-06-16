#!/usr/bin/env python3
"""
Air-Gapped NOC Copilot - Main Training Script
Trains and evaluates LSTM/TCN models for predictive network fault detection.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

from data_preprocessor import (
    NetworkDataPreprocessor,
    PreprocessingConfig,
    SyntheticNetworkDataGenerator,
)
from lstm_model import LSTMPredictor, LSTMMultiTask, LSTMTrainer, create_model_config
from tcn_model import TCNPredictor, TCNLSTMHybrid, TCNTrainer, create_tcn_config, create_hybrid_config


# ════════════════════════════════════════════════════════════════
# Model Evaluator
# ════════════════════════════════════════════════════════════════

class ModelEvaluator:
    """Evaluates model performance with network-specific metrics."""
    
    def __init__(self):
        self.results = {}
    
    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: str = "cpu",
    ) -> Dict:
        """Evaluate model on test set."""
        model.eval()
        all_predictions = []
        all_targets = []
        
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            predictions = model(batch_X)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        metrics = self._compute_metrics(predictions, targets)
        
        return metrics
    
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Dict:
        """Compute comprehensive evaluation metrics."""
        # Flatten for comparison
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Basic metrics
        mse = float(np.mean((pred_flat - target_flat) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(pred_flat - target_flat)))
        mape = float(np.mean(np.abs((target_flat - pred_flat) / (target_flat + 1e-6))) * 100)
        
        # R-squared
        ss_res = np.sum((target_flat - pred_flat) ** 2)
        ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
        r_squared = float(1 - (ss_res / (ss_tot + 1e-6)))
        
        # Correlation
        correlation = float(np.corrcoef(pred_flat, target_flat)[0, 1])
        
        # Network-specific: Prediction lead time accuracy
        # (how early can we predict threshold breach?)
        threshold = 90.0
        pred_breach = pred_flat > threshold
        actual_breach = target_flat > threshold
        
        # True positive rate for breach detection
        true_positive = np.sum(pred_breach & actual_breach)
        false_positive = np.sum(pred_breach & ~actual_breach)
        false_negative = np.sum(~pred_breach & actual_breach)
        
        precision = true_positive / (true_positive + false_positive + 1e-6)
        recall = true_positive / (true_positive + false_negative + 1e-6)
        f1_score = 2 * precision * recall / (precision + recall + 1e-6)
        
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r_squared": r_squared,
            "correlation": correlation,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "num_samples": len(pred_flat),
        }
        
        return metrics
    
    def print_report(self, metrics: Dict, model_name: str):
        """Print formatted evaluation report."""
        print(f"\n{'='*60}")
        print(f"  {model_name} Evaluation Report")
        print(f"{'='*60}\n")
        
        print("Regression Metrics:")
        print(f"  MSE:             {metrics['mse']:.6f}")
        print(f"  RMSE:            {metrics['rmse']:.6f}")
        print(f"  MAE:             {metrics['mae']:.6f}")
        print(f"  MAPE:            {metrics['mape']:.2f}%")
        print(f"  R-squared:       {metrics['r_squared']:.4f}")
        print(f"  Correlation:     {metrics['correlation']:.4f}")
        
        print("\nBreach Detection Metrics:")
        print(f"  Precision:       {metrics['precision']:.4f}")
        print(f"  Recall:          {metrics['recall']:.4f}")
        print(f"  F1 Score:        {metrics['f1_score']:.4f}")
        
        print(f"\nTest Samples:    {metrics['num_samples']}")
        print(f"{'='*60}\n")


# ════════════════════════════════════════════════════════════════
# Main Training Pipeline
# ════════════════════════════════════════════════════════════════

class TrainingPipeline:
    """Complete training pipeline for network fault prediction models."""
    
    def __init__(self, data_path: str, output_dir: str = "models"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
    
    def preprocess_data(self) -> Tuple:
        """Preprocess the telemetry data."""
        config = PreprocessingConfig(
            sequence_length=30,
            forecast_horizon=5,
            batch_size=32,
        )
        
        preprocessor = NetworkDataPreprocessor(config)
        train_loader, val_loader, test_loader, metadata = preprocessor.process_pipeline(
            self.data_path
        )
        
        # Save preprocessor state
        preprocessor.save_scalers(str(self.output_dir / "scalers.pkl"))
        
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return train_loader, val_loader, test_loader, metadata
    
    def train_lstm(self, train_loader, val_loader, test_loader, metadata) -> Dict:
        """Train LSTM model."""
        print("\n" + "="*60)
        print("  Training LSTM Model")
        print("="*60)
        
        config = create_model_config(
            input_size=metadata["num_features"],
            forecast_horizon=metadata["forecast_horizon"],
        )
        
        model = LSTMPredictor(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            output_size=config["output_size"],
            dropout=config["dropout"],
            bidirectional=config["bidirectional"],
            forecast_horizon=config["forecast_horizon"],
        )
        
        trainer = LSTMTrainer(model, config, self.device)
        history = trainer.train(
            train_loader,
            val_loader,
            num_epochs=config["num_epochs"],
            patience=config["patience"],
        )
        
        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, test_loader, self.device)
        evaluator.print_report(metrics, "LSTM")
        
        # Save model
        torch.save(model.state_dict(), self.output_dir / "lstm_model.pth")
        
        return {"history": history, "metrics": metrics, "config": config}
    
    def train_tcn(self, train_loader, val_loader, test_loader, metadata) -> Dict:
        """Train TCN model."""
        print("\n" + "="*60)
        print("  Training TCN Model")
        print("="*60)
        
        config = create_tcn_config(
            input_size=metadata["num_features"],
            forecast_horizon=metadata["forecast_horizon"],
        )
        
        model = TCNPredictor(
            input_size=config["input_size"],
            num_channels=config["num_channels"],
            kernel_size=config["kernel_size"],
            dropout=config["dropout"],
            forecast_horizon=config["forecast_horizon"],
        )
        
        trainer = TCNTrainer(model, config, self.device)
        history = trainer.train(
            train_loader,
            val_loader,
            num_epochs=config["num_epochs"],
            patience=config["patience"],
        )
        
        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, test_loader, self.device)
        evaluator.print_report(metrics, "TCN")
        
        # Save model
        torch.save(model.state_dict(), self.output_dir / "tcn_model.pth")
        
        return {"history": history, "metrics": metrics, "config": config}
    
    def train_hybrid(self, train_loader, val_loader, test_loader, metadata) -> Dict:
        """Train hybrid TCN-LSTM model."""
        print("\n" + "="*60)
        print("  Training Hybrid TCN-LSTM Model")
        print("="*60)
        
        config = create_hybrid_config(
            input_size=metadata["num_features"],
            forecast_horizon=metadata["forecast_horizon"],
        )
        
        model = TCNLSTMHybrid(
            input_size=config["input_size"],
            tcn_channels=config["tcn_channels"],
            lstm_hidden=config["lstm_hidden"],
            lstm_layers=config["lstm_layers"],
            dropout=config["dropout"],
            forecast_horizon=config["forecast_horizon"],
        )
        
        trainer = TCNTrainer(model, config, self.device)
        history = trainer.train(
            train_loader,
            val_loader,
            num_epochs=config["num_epochs"],
            patience=config["patience"],
        )
        
        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, test_loader, self.device)
        evaluator.print_report(metrics, "Hybrid TCN-LSTM")
        
        # Save model
        torch.save(model.state_dict(), self.output_dir / "hybrid_model.pth")
        
        return {"history": history, "metrics": metrics, "config": config}
    
    def run(self, models: list = None):
        """Run the complete training pipeline."""
        if models is None:
            models = ["lstm", "tcn", "hybrid"]
        
        print("\n" + "="*60)
        print("  Air-Gapped NOC Copilot - Model Training Pipeline")
        print("="*60)
        
        # Generate synthetic data if needed
        if not os.path.exists(self.data_path):
            print("\nGenerating synthetic training data...")
            generator = SyntheticNetworkDataGenerator(num_samples=10000)
            df = generator.generate()
            df.to_csv(self.data_path, index=False)
            print(f"Saved {len(df)} records to {self.data_path}")
        
        # Preprocess data
        print("\n[Step 1/4] Preprocessing data...")
        train_loader, val_loader, test_loader, metadata = self.preprocess_data()
        
        # Train models
        results = {}
        
        if "lstm" in models:
            print("\n[Step 2/4] Training LSTM...")
            results["lstm"] = self.train_lstm(
                train_loader, val_loader, test_loader, metadata
            )
        
        if "tcn" in models:
            print("\n[Step 3/4] Training TCN...")
            results["tcn"] = self.train_tcn(
                train_loader, val_loader, test_loader, metadata
            )
        
        if "hybrid" in models:
            print("\n[Step 4/4] Training Hybrid...")
            results["hybrid"] = self.train_hybrid(
                train_loader, val_loader, test_loader, metadata
            )
        
        # Save results summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "device": self.device,
            "metadata": metadata,
            "models_trained": list(results.keys()),
            "model_results": {
                name: {
                    "best_val_loss": min(result["history"]["val_loss"]),
                    "metrics": result["metrics"],
                }
                for name, result in results.items()
            },
        }
        
        with open(self.output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Print comparison
        print("\n" + "="*60)
        print("  Model Comparison Summary")
        print("="*60)
        
        for name, result in results.items():
            metrics = result["metrics"]
            val_loss = min(result["history"]["val_loss"])
            print(f"\n{name.upper()}:")
            print(f"  Best Val Loss:  {val_loss:.4f}")
            print(f"  RMSE:           {metrics['rmse']:.4f}")
            print(f"  MAE:            {metrics['mae']:.4f}")
            print(f"  R-squared:      {metrics['r_squared']:.4f}")
            print(f"  F1 Score:       {metrics['f1_score']:.4f}")
        
        print(f"\n{'='*60}")
        print(f"All models saved to: {self.output_dir}")
        print(f"{'='*60}\n")
        
        return results


# ════════════════════════════════════════════════════════════════
# CLI Entry Point
# ════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train predictive models for network fault detection"
    )
    parser.add_argument(
        "--data", type=str, default="synthetic_telemetry.csv",
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--output", type=str, default="models",
        help="Output directory for trained models"
    )
    parser.add_argument(
        "--models", nargs="+", default=["lstm", "tcn", "hybrid"],
        choices=["lstm", "tcn", "hybrid"],
        help="Models to train"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    pipeline = TrainingPipeline(
        data_path=args.data,
        output_dir=args.output,
    )
    
    results = pipeline.run(models=args.models)
