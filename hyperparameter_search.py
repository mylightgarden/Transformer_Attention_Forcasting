'''
Author: Sophie Zhao
'''

import itertools
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from stock_pct_change_transformer import train_model, evaluate_model, set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameter Search for SimpleTransformer")
    parser.add_argument("--csv", required=True, help="models/transformer/spy_features.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.15, help="Test split fraction")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation split fraction of remaining")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    set_seed(args.seed)
    df = pd.read_csv(args.csv, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    features = [c for c in df.columns if c not in ["Date","Close","PctChange"]]
    X = df[features].values
    y = df["PctChange"].values

    # Global scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into train+val/test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size, shuffle=False
    )
    # Further split train+val into train/val
    val_frac = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, shuffle=False
    )

    # Define hyperparameter grid
    grid = {
        'seq_len': [14],
        'd_model': [32, 64],
        'nhead':   [4],
        'num_layers': [1, 2, 3],
        'dropout': [0.0, 0.1, 0.2],
        'batch':   [16, 32],
        'lr':      [1e-3, 5e-4, 1e-4]
    }

    # Flatten grid
    keys, values = zip(*grid.items())
    best = {'params': None, 'val_loss': float('inf')}

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        print(f"Testing: {params}")
        try:
            _, val_loss, _ = train_model(
                X_train, y_train, X_val, y_val,
                feat_dim=X_train.shape[1],
                seq_len=params['seq_len'],
                device=torch.device('cpu'),
                epochs=args.epochs,
                batch=params['batch'],
                lr=params['lr'],
                d_model=params['d_model'],
                nhead=params['nhead'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                patience=5
            )
        except Exception as e:
            print(f"  Skipped due to error: {e}")
            continue
        print(f"  -> val_loss = {val_loss:.6f}\n")
        if val_loss < best['val_loss']:
            best['val_loss'] = val_loss
            best['params'] = params.copy()

    print("Best hyperparameters:", best['params'])
    print("Best validation loss:", best['val_loss'])

    # Evaluate best on test
    print("Evaluating best on test set...")
    _, _, model = train_model(
        X_trainval, y_trainval, X_test, y_test,
        feat_dim=X_trainval.shape[1],
        seq_len=best['params']['seq_len'],
        device=torch.device('cpu'),
        epochs=args.epochs,
        batch=best['params']['batch'],
        lr=best['params']['lr'],
        d_model=best['params']['d_model'],
        nhead=best['params']['nhead'],
        num_layers=best['params']['num_layers'],
        dropout=best['params']['dropout'],
        patience=5
    )
    evaluate_model(model, X_test, y_test,
                   seq_len=best['params']['seq_len'],
                   device=torch.device('cpu'),
                   batch=best['params']['batch'])
