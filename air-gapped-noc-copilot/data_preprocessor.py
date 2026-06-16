#!/usr/bin/env python3
"""
Air-Gapped NOC Copilot - Data Preprocessing Pipeline
Preprocesses network telemetry data for LSTM/TCN training.
Handles feature engineering, normalization, and sequence creation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


# ════════════════════════════════════════════════════════════════
# Data Configuration
# ════════════════════════════════════════════════════════════════

@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing pipeline."""
    sequence_length: int = 60  # Lookback window (timesteps)
    forecast_horizon: int = 10  # Prediction horizon (timesteps)
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 32
    num_features: int = 25
    scaler_type: str = "robust"  # "standard", "robust", "minmax"
    handle_missing: str = "interpolate"  # "interpolate", "forward_fill", "drop"
    feature_columns: List[str] = None

    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [
                # Interface metrics
                "interface_utilization",
                "interface_in_errors",
                "interface_out_errors",
                "interface_in_discards",
                "interface_out_discards",
                "interface_in_packets",
                "interface_out_packets",
                # Latency metrics
                "latency_avg_ms",
                "latency_jitter_ms",
                "packet_loss_percent",
                # Routing metrics
                "ospf_neighbor_state",
                "ospf_dead_timer",
                "bgp_fsm_transitions",
                "bgp_update_rate",
                "mpls_ldp_state_changes",
                # IPSec metrics
                "ipsec_throughput_bps",
                "ipsec_error_rate",
                # Congestion indicators
                "congestion_trend",
                "queue_depth",
                "tti_estimate",
                # Time features
                "hour_of_day",
                "day_of_week",
                "is_peak_hour",
                # System metrics
                "cpu_utilization",
                "memory_utilization",
            ]


# ════════════════════════════════════════════════════════════════
# Feature Scalers
# ════════════════════════════════════════════════════════════════

class RobustScaler:
    """RobustScaler using median and IQR (resistant to outliers)."""
    
    def __init__(self):
        self.median_ = None
        self.iqr_ = None
        self.scale_ = None
    
    def fit(self, X: np.ndarray) -> 'RobustScaler':
        self.median_ = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        self.iqr_ = q75 - q25
        self.iqr_[self.iqr_ == 0] = 1.0
        self.scale_ = self.iqr_
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.median_) / self.scale_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.scale_ + self.median_


class StandardScaler:
    """StandardScaler using mean and std."""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X: np.ndarray) -> 'StandardScaler':
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std_ + self.mean_


class MinMaxScaler:
    """MinMaxScaler to [0, 1] range."""
    
    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.scale_ = None
    
    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.scale_ = self.max_ - self.min_
        self.scale_[self.scale_ == 0] = 1.0
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / self.scale_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.scale_ + self.min_


# ════════════════════════════════════════════════════════════════
# Feature Engineering
# ════════════════════════════════════════════════════════════════

class NetworkFeatureEngineer:
    """Engineers features from raw network telemetry data."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    
    def create_utilization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create utilization-based features."""
        # Moving averages
        for window in [5, 10, 30]:
            df[f'util_ma_{window}'] = df['interface_utilization'].rolling(
                window=window, min_periods=1
            ).mean()
        
        # Rate of change
        df['util_roc_5'] = df['interface_utilization'].diff(5) / 5
        df['util_roc_10'] = df['interface_utilization'].diff(10) / 10
        
        # Volatility
        df['util_volatility'] = df['interface_utilization'].rolling(
            window=10, min_periods=1
        ).std()
        
        # Threshold proximity
        df['util_threshold_proximity'] = 100 - df['interface_utilization']
        
        return df
    
    def create_latency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create latency-based features."""
        # Moving averages
        df['latency_ma_5'] = df['latency_avg_ms'].rolling(window=5, min_periods=1).mean()
        df['latency_ma_15'] = df['latency_avg_ms'].rolling(window=15, min_periods=1).mean()
        
        # Latency variance
        df['latency_var'] = df['latency_avg_ms'].rolling(window=10, min_periods=1).var()
        
        # Latency trend
        df['latency_trend'] = df['latency_ma_5'] - df['latency_ma_15']
        
        # Jitter ratio
        df['jitter_ratio'] = df['latency_jitter_ms'] / (df['latency_avg_ms'] + 1e-6)
        
        return df
    
    def create_routing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create routing protocol features."""
        # OSPF stability score
        df['ospf_stability'] = 1.0 - (df['ospf_neighbor_state'] / 8.0)
        
        # BGP churn rate
        df['bgp_churn'] = df['bgp_fsm_transitions'] / (df['bgp_update_rate'] + 1)
        
        # Routing change rate
        df['routing_change_rate'] = (
            df['ospf_dead_timer'].diff().abs() +
            df['mpls_ldp_state_changes'].diff().abs()
        )
        
        return df
    
    def create_ipsec_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create IPSec tunnel health features."""
        # Throughput utilization
        df['ipsec_throughput_pct'] = (
            df['ipsec_throughput_bps'] / 
            (df['interface_utilization'] * 1e6 + 1e-6) * 100
        )
        
        # Error ratio
        df['ipsec_error_ratio'] = (
            df['ipsec_error_rate'] / (df['ipsec_throughput_bps'] + 1e-6)
        )
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour_of_day'] = df['timestamp'].dt.hour / 24.0
            df['day_of_week'] = df['timestamp'].dt.dayofweek / 7.0
            df['is_peak_hour'] = (
                (df['timestamp'].dt.hour >= 9) & 
                (df['timestamp'].dt.hour <= 17)
            ).astype(float)
        return df
    
    def create_congestion_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create congestion prediction features."""
        # Congestion buildup rate
        df['congestion_buildup'] = (
            df['interface_utilization'].diff(5) / 5
        )
        
        # Projected time-to-impact
        df['projected_tti'] = (
            (100 - df['interface_utilization']) / 
            (df['congestion_buildup'].abs() + 1e-6)
        )
        
        # Queue pressure
        df['queue_pressure'] = (
            df['interface_out_discards'] / 
            (df['interface_out_packets'] + 1e-6)
        )
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        df = self.create_utilization_features(df)
        df = self.create_latency_features(df)
        df = self.create_routing_features(df)
        df = self.create_ipsec_features(df)
        df = self.create_temporal_features(df)
        df = self.create_congestion_indicators(df)
        return df


# ════════════════════════════════════════════════════════════════
# Data Preprocessing Pipeline
# ════════════════════════════════════════════════════════════════

class NetworkDataPreprocessor:
    """Complete preprocessing pipeline for network telemetry data."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.feature_engineer = NetworkFeatureEngineer(config)
        self.scalers: Dict[str, object] = {}
        self.feature_names: List[str] = []
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load telemetry data from CSV or Parquet."""
        path = Path(filepath)
        if path.suffix == '.csv':
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
        elif path.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        if self.config.handle_missing == "interpolate":
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
                df = df.interpolate(method="time", limit=5)
                df = df.reset_index()
            else:
                df = df.interpolate(method="linear", limit=5)
        elif self.config.handle_missing == "forward_fill":
            df = df.fillna(method="ffill", limit=5)
        elif self.config.handle_missing == "drop":
            df = df.dropna()

        # Fill any remaining NaN with 0
        df = df.fillna(0)

        return df
    
    def create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for time series forecasting."""
        sequences_X = []
        sequences_y = []
        
        for i in range(len(X) - self.config.sequence_length - self.config.forecast_horizon + 1):
            seq_X = X[i:i + self.config.sequence_length]
            seq_y = y[i + self.config.sequence_length:
                       i + self.config.sequence_length + self.config.forecast_horizon]
            sequences_X.append(seq_X)
            sequences_y.append(seq_y)
        
        return np.array(sequences_X), np.array(sequences_y)
    
    def create_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Create prediction labels from telemetry data."""
        # Label: Will utilization exceed 90% in next forecast_horizon steps?
        future_max_util = df['interface_utilization'].rolling(
            window=self.config.forecast_horizon, min_periods=1
        ).max().shift(-self.config.forecast_horizon)
        
        labels = (future_max_util > 90).astype(float)
        return labels.values
    
    def fit_scalers(self, X: np.ndarray):
        """Fit scalers on training data."""
        X_2d = X.reshape(-1, X.shape[-1])
        
        scaler_map = {
            "standard": StandardScaler,
            "robust": RobustScaler,
            "minmax": MinMaxScaler,
        }
        
        ScalerClass = scaler_map.get(self.config.scaler_type, RobustScaler)
        
        self.scalers['feature'] = ScalerClass()
        self.scalers['feature'].fit(X_2d)
        
        self.scalers['label'] = StandardScaler()
    
    def transform_data(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scalers."""
        original_shape = X.shape
        X_2d = X.reshape(-1, X.shape[-1])
        X_scaled = self.scalers['feature'].transform(X_2d)
        return X_scaled.reshape(original_shape)
    
    def inverse_transform_labels(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform labels."""
        if 'label' in self.scalers:
            original_shape = y.shape
            y_2d = y.reshape(-1, y.shape[-1] if len(y.shape) > 1 else 1)
            y_inv = self.scalers['label'].inverse_transform(y_2d)
            return y_inv.reshape(original_shape)
        return y
    
    def split_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Tuple, Tuple, Tuple]:
        """Split data into train, validation, and test sets."""
        n = len(X)
        train_end = int(n * self.config.train_split)
        val_end = int(n * (self.config.train_split + self.config.val_split))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def prepare_dataloaders(
        self, 
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch DataLoaders."""
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def save_scalers(self, filepath: str):
        """Save fitted scalers to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scalers, f)
    
    def load_scalers(self, filepath: str):
        """Load fitted scalers from disk."""
        with open(filepath, 'rb') as f:
            self.scalers = pickle.load(f)
    
    def process_pipeline(
        self, filepath: str
    ) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
        """Complete preprocessing pipeline."""
        # Load data
        print(f"[1/7] Loading data from {filepath}...")
        df = self.load_data(filepath)
        print(f"  Loaded {len(df)} records, {len(df.columns)} columns")
        
        # Handle missing values
        print("[2/7] Handling missing values...")
        df = self.handle_missing_values(df)
        
        # Engineer features
        print("[3/7] Engineering features...")
        df = self.feature_engineer.engineer_all_features(df)
        print(f"  Created {len(df.columns)} total features")
        
        # Select feature columns
        available_features = [
            f for f in self.config.feature_columns if f in df.columns
        ]
        self.feature_names = available_features
        print(f"  Using {len(available_features)} features")
        
        # Create sequences
        print("[4/7] Creating sequences...")
        X = df[available_features].values
        y = self.create_labels(df)
        
        X_seq, y_seq = self.create_sequences(X, y)
        print(f"  Created {len(X_seq)} sequences of length {self.config.sequence_length}")
        
        # Split data
        print("[5/7] Splitting data...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.split_data(X_seq, y_seq)
        print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Fit scalers and transform
        print("[6/7] Fitting scalers and transforming...")
        self.fit_scalers(X_train)
        X_train_scaled = self.transform_data(X_train)
        X_val_scaled = self.transform_data(X_val)
        X_test_scaled = self.transform_data(X_test)
        
        # Create dataloaders
        print("[7/7] Creating dataloaders...")
        train_loader, val_loader, test_loader = self.prepare_dataloaders(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            X_test_scaled, y_test
        )
        
        metadata = {
            "num_features": len(available_features),
            "feature_names": available_features,
            "sequence_length": self.config.sequence_length,
            "forecast_horizon": self.config.forecast_horizon,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "scaler_type": self.config.scaler_type,
        }
        
        print("\n[OK] Preprocessing pipeline complete!")
        return train_loader, val_loader, test_loader, metadata


# ════════════════════════════════════════════════════════════════
# Synthetic Data Generator (for testing)
# ════════════════════════════════════════════════════════════════

class SyntheticNetworkDataGenerator:
    """Generates synthetic network telemetry data for testing."""
    
    def __init__(self, num_samples: int = 10000):
        self.num_samples = num_samples
    
    def generate(self) -> pd.DataFrame:
        """Generate synthetic network telemetry data."""
        np.random.seed(42)
        
        timestamps = pd.date_range(
            start='2024-01-01', periods=self.num_samples, freq='10s'
        )
        
        # Base utilization pattern (daily cycle)
        hours = np.arange(self.num_samples) / 360  # Convert to hours
        base_util = 40 + 20 * np.sin(2 * np.pi * hours / 24) + 10 * np.sin(2 * np.pi * hours / 168)
        
        # Add noise and occasional spikes
        util_noise = np.random.normal(0, 5, self.num_samples)
        util_spikes = np.random.choice([0, 50, 0], size=self.num_samples, p=[0.95, 0.05, 0])
        utilization = np.clip(base_util + util_noise + util_spikes, 0, 100)
        
        # Inject faults (congestion buildup, BGP flaps, etc.)
        fault_indices = np.random.choice(self.num_samples, size=100, replace=False)
        for idx in fault_indices:
            if idx + 50 < self.num_samples:
                utilization[idx:idx+50] += np.linspace(0, 60, 50)
        
        # Generate correlated metrics
        data = {
            'timestamp': timestamps,
            'interface_utilization': utilization,
            'interface_in_errors': np.random.poisson(0.1, self.num_samples).astype(float),
            'interface_out_errors': np.random.poisson(0.05, self.num_samples).astype(float),
            'interface_in_discards': np.random.poisson(0.2, self.num_samples).astype(float),
            'interface_out_discards': np.random.poisson(0.1, self.num_samples).astype(float),
            'interface_in_packets': np.random.exponential(1000, self.num_samples),
            'interface_out_packets': np.random.exponential(800, self.num_samples),
            'latency_avg_ms': 10 + 5 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 2, self.num_samples),
            'latency_jitter_ms': np.abs(np.random.normal(1, 0.5, self.num_samples)),
            'packet_loss_percent': np.clip(np.random.exponential(0.1, self.num_samples), 0, 5),
            'ospf_neighbor_state': np.where(
                utilization > 85, 
                np.random.choice([8, 7, 6], self.num_samples, p=[0.6, 0.3, 0.1]),
                8
            ).astype(float),
            'ospf_dead_timer': np.full(self.num_samples, 40.0) + np.random.normal(0, 1, self.num_samples),
            'bgp_fsm_transitions': np.random.poisson(0.01, self.num_samples).astype(float),
            'bgp_update_rate': np.random.exponential(10, self.num_samples),
            'mpls_ldp_state_changes': np.random.poisson(0.005, self.num_samples).astype(float),
            'ipsec_throughput_bps': np.random.exponential(5e6, self.num_samples),
            'ipsec_error_rate': np.random.exponential(0.001, self.num_samples),
            'congestion_trend': np.random.normal(0, 1, self.num_samples),
            'queue_depth': np.random.exponential(0.1, self.num_samples),
            'tti_estimate': np.full(self.num_samples, 100.0) - utilization + np.random.normal(0, 5, self.num_samples),
            'hour_of_day': (timestamps.hour + timestamps.minute / 60) / 24,
            'day_of_week': timestamps.dayofweek / 7,
            'is_peak_hour': ((timestamps.hour >= 9) & (timestamps.hour <= 17)).astype(float),
            'cpu_utilization': 30 + 10 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 5, self.num_samples),
            'memory_utilization': 60 + 10 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 3, self.num_samples),
        }
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    # Test the preprocessor with synthetic data
    print("Generating synthetic data...")
    generator = SyntheticNetworkDataGenerator(num_samples=5000)
    df = generator.generate()
    
    # Save to CSV
    df.to_csv("synthetic_telemetry.csv", index=False)
    print(f"Saved {len(df)} records to synthetic_telemetry.csv")
    
    # Test preprocessing
    config = PreprocessingConfig(sequence_length=30, forecast_horizon=5)
    preprocessor = NetworkDataPreprocessor(config)
    
    train_loader, val_loader, test_loader, metadata = preprocessor.process_pipeline(
        "synthetic_telemetry.csv"
    )
    
    print(f"\nMetadata: {metadata}")
