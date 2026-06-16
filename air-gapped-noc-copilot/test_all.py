#!/usr/bin/env python3
"""Comprehensive test script for all components."""
import sys
import traceback

def test_routing_config():
    print("[1/6] Testing routing configuration generation...")
    from configure_routing import TOPOLOGY, generate_frr_config
    for name, node in TOPOLOGY.items():
        cfg = generate_frr_config(name, node)
        assert "router ospf" in cfg or node["role"] == "CE", f"Missing OSPF in {name}"
        assert "mpls ldp" in cfg or node["role"] == "CE", f"Missing MPLS in {name}"
    print(f"  PASS: {len(TOPOLOGY)} node configs validated")

def test_ipsec_config():
    print("[2/6] Testing IPSec overlay configuration...")
    from configure_ipsec_overlay import SDWAN_SITES, generate_full_ipsec_config
    for name, site in SDWAN_SITES.items():
        cfg = generate_full_ipsec_config(name, site)
        assert "crypto ikev2" in cfg or site.role == "hub", f"Missing IPSec in {name}"
    print(f"  PASS: {len(SDWAN_SITES)} IPSec configs validated")

def test_preprocessor():
    print("[3/6] Testing data preprocessor...")
    import os
    from data_preprocessor import NetworkDataPreprocessor, PreprocessingConfig, SyntheticNetworkDataGenerator

    # Generate synthetic data
    gen = SyntheticNetworkDataGenerator(num_samples=2000)
    df = gen.generate()
    assert len(df) == 2000, f"Expected 2000 rows, got {len(df)}"
    assert "interface_utilization" in df.columns

    # Save to CSV
    csv_path = "_test_telemetry.csv"
    df.to_csv(csv_path, index=False)

    # Run the full pipeline
    config = PreprocessingConfig(sequence_length=30, forecast_horizon=5)
    preprocessor = NetworkDataPreprocessor(config)
    train_loader, val_loader, test_loader, meta = preprocessor.process_pipeline(csv_path)

    assert meta["num_features"] > 0
    assert meta["train_size"] > 0
    assert meta["val_size"] > 0
    assert meta["test_size"] > 0

    # Cleanup
    os.remove(csv_path)
    print(f"  PASS: Pipeline complete. Train={meta['train_size']}, Val={meta['val_size']}, Test={meta['test_size']}")

def test_lstm_model():
    print("[4/6] Testing LSTM model...")
    import torch
    import numpy as np
    from lstm_model import LSTMPredictor, LSTMMultiTask, LSTMInferenceEngine, create_model_config

    config = create_model_config(25, 10)
    model = LSTMPredictor(25, 128, 2, 1, 0.2, False, 10)
    x = torch.randn(4, 30, 25)
    out = model(x)
    assert out.shape == (4, 10, 1), f"LSTM shape mismatch: {out.shape}"

    multi = LSTMMultiTask(25, 128, 2, 10, 0.2)
    out_m = multi(x)
    assert out_m["forecast"].shape == (4, 10, 1)
    assert out_m["anomaly_prob"].shape == (4, 10)

    engine = LSTMInferenceEngine(model)
    result = engine.detect_anomaly(np.random.randn(30, 25).astype(np.float32))
    assert "is_anomaly" in result
    print(f"  PASS: LSTM forward, multi-task, inference all OK")

def test_tcn_model():
    print("[5/6] Testing TCN model...")
    import torch
    from tcn_model import TCNPredictor, TCNLSTMHybrid, TCNPredictorEngine, create_tcn_config, create_hybrid_config

    tcn = TCNPredictor(25, [64, 128, 128, 64], 3, 0.2, 1, 10)
    x = torch.randn(4, 30, 25)
    out = tcn(x)
    assert out.shape == (4, 10, 1), f"TCN shape mismatch: {out.shape}"

    hyb = TCNLSTMHybrid(25, [64, 64, 128], 128, 2, 0.2, 1, 10)
    out_h = hyb(x)
    assert out_h.shape == (4, 10, 1), f"Hybrid shape mismatch: {out_h.shape}"

    engine = TCNPredictorEngine(tcn)
    pred = engine.predict(x)
    assert pred.shape == (4, 10, 1)
    print(f"  PASS: TCN forward, hybrid, inference all OK")

def test_training_loop():
    print("[6/6] Testing training loop...")
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from lstm_model import LSTMPredictor, LSTMTrainer, create_model_config

    config = create_model_config(25, 5)
    config["num_epochs"] = 3
    config["patience"] = 2
    model = LSTMPredictor(25, 32, 1, 1, 0.1, False, 5)

    X_train = torch.randn(80, 30, 25)
    y_train = torch.randn(80, 5, 1)
    X_val = torch.randn(20, 30, 25)
    y_val = torch.randn(20, 5, 1)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=16)
    val_loader = DataLoader(val_ds, batch_size=16)

    trainer = LSTMTrainer(model, config, "cpu")
    history = trainer.train(train_loader, val_loader, num_epochs=3, patience=2)
    assert len(history["train_loss"]) > 0
    assert len(history["val_loss"]) > 0
    print(f"  PASS: Training ran for {len(history['train_loss'])} epochs")

if __name__ == "__main__":
    errors = []
    tests = [
        test_routing_config,
        test_ipsec_config,
        test_preprocessor,
        test_lstm_model,
        test_tcn_model,
        test_training_loop,
    ]
    for test in tests:
        try:
            test()
        except Exception as e:
            errors.append((test.__name__, str(e)))
            traceback.print_exc()
    print()
    if errors:
        print(f"FAILED: {len(errors)} test(s)")
        for name, err in errors:
            print(f"  {name}: {err}")
        sys.exit(1)
    else:
        print("ALL 6 TESTS PASSED")
