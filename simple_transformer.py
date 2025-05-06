'''
Author: Sophie Zhao
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


class SimpleTransformer(nn.Module):
    def __init__(self, feat_dim, seq_len, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(feat_dim, d_model)

        # Positional encoding
        self.register_buffer('pos_embedding', self._get_pos_embedding(seq_len, d_model))

        # Transformer encoder
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model,
        #     nhead=nhead,
        #     dim_feedforward=d_model * 4,
        #     dropout=dropout,
        #     batch_first=True
        # )
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layers = nn.ModuleList([
            FinancialTransformerLayer(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

        # Output projection for mean and std
        self.mean_proj = nn.Linear(d_model, 1)
        self.std_proj = nn.Linear(d_model, 1)

        # Initialize weights
        self._init_weights()

    def _get_pos_embedding(self, seq_len, d_model):
        pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_embedding = torch.zeros(1, seq_len, d_model)
        pos_embedding[0, :, 0::2] = torch.sin(pos * div_term)
        pos_embedding[0, :, 1::2] = torch.cos(pos * div_term)
        return pos_embedding

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x shape: (batch_size, seq_len, feat_dim)
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)

        # Add positional embedding
        x = x + self.pos_embedding

        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x)
        # back to (batch, seq, d_model)
        x = x.transpose(0, 1)

        # 6) Pool & project
        last = x[:, -1, :]  # (batch, d_model)
        mean = self.mean_proj(last)  # (batch, 1)
        std = F.softplus(self.std_proj(last)) + 1e-6

        return mean, std


class FinancialTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=False,
            batch_first=True   # use batch_first to skip transposes
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * 2) # Extend model dimension
        self.linear2 = nn.Linear(d_model * 2, d_model) # Move back to original model dimension
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq, d_model)
        attn_out, _ = self.self_attn(x, x, x)   # now batch_first=True
        x = x + self.dropout(attn_out)
        # x = self.norm1(x)

        ffn = F.gelu(self.linear1(x))
        x = self.norm1(x)
        ffn = self.dropout(self.linear2(ffn))
        x = x + ffn
        x = self.norm2(x)

        return x



class FinancialLoss(nn.Module):
    """Combines adaptive Huber loss with directional penalty"""

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, preds, targets):
        pred_mean, pred_std = preds

        # Adaptive Huber loss
        error = targets - pred_mean
        delta = pred_std.detach().clamp(min=0.1, max=2.0)
        huber_loss = torch.where(
            error.abs() < delta,
            0.5 * error.pow(2),
            delta * (error.abs() - 0.5 * delta)
        )

        # Directional penalty
        sign_loss = F.relu(-torch.sign(pred_mean) * targets)  # Only penalize wrong signs

        return self.alpha * huber_loss.mean() + (1 - self.alpha) * sign_loss.mean()




def train_model(X_train, y_train, X_val, y_val, feat_dim, seq_len, device, epochs, batch, lr,
                d_model=64, nhead=4, num_layers=3, dropout=0.1, patience=10, weight_decay=1e-4):
    """Train the model on the given data."""

    # Create sequences from the data
    def create_sequences(X, y, overlap=True):
        Xs, ys = [], []
        if overlap:
            # Overlapping sequences for training (more data)
            for i in range(len(X) - seq_len + 1):
                Xs.append(X[i:i + seq_len])
                ys.append(y[i + seq_len - 1])
        else:
            # Non-overlapping sequences for validation/test (no data leakage)
            for i in range(0, len(X) - seq_len + 1, seq_len):
                Xs.append(X[i:i + seq_len])
                ys.append(y[i + seq_len - 1])
        return np.array(Xs), np.array(ys)

    # Create sequences - overlapping for training, non-overlapping for validation
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, overlap=True)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, overlap=False)

    # Create tensors
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    # Create model with specified parameters
    model = SimpleTransformer(
        feat_dim=feat_dim,
        seq_len=seq_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = FinancialLoss(alpha=0.5)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad() # resets the gradients of the model's parameters to zero
            pred_mean, pred_std = model(xb)
            loss = criterion((pred_mean.squeeze(), pred_std.squeeze()), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred_mean, pred_std = model(xb)
                val_loss += criterion((pred_mean.squeeze(), pred_std.squeeze()), yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return best_model_state, best_val_loss, model


# def evaluate_model(model, X_test, y_test, seq_len, device, batch):
#     """Evaluate the model on the test data."""
#
#     def create_sequences(X, y, overlap=False):
#         Xs, ys = [], []
#         step = 1 if overlap else seq_len
#         for i in range(0, len(X) - seq_len + 1, step):
#             Xs.append(X[i:i + seq_len])
#             ys.append(y[i + seq_len - 1])
#         return np.array(Xs), np.array(ys)
#
#     # Create sequences and tensors
#     X_test_seq, y_test_seq = create_sequences(X_test, y_test)
#     test_loader = DataLoader(
#         TensorDataset(
#             torch.tensor(X_test_seq, dtype=torch.float32),
#             torch.tensor(y_test_seq, dtype=torch.float32)
#         ),
#         batch_size=batch,
#         shuffle=False
#     )
#
#     model.eval()
#     predictions, targets = [], []
#
#     with torch.no_grad():
#         for xb, yb in test_loader:
#             xb, yb = xb.to(device), yb.to(device)
#             pred_mean, _ = model(xb)  # We only need mean for point prediction metrics
#             predictions.append(pred_mean.squeeze().cpu().numpy())
#             targets.append(yb.cpu().numpy())
#
#     predictions = np.concatenate(predictions)
#     targets = np.concatenate(targets)
#
#     # Calculate all metrics
#     mse = np.mean((predictions - targets) ** 2)
#     mae = np.mean(np.abs(predictions - targets))
#     rmse = np.sqrt(mse)
#     directional_acc = np.mean(np.sign(predictions) == np.sign(targets)) * 100
#     r2 = r2_score(targets, predictions)
#
#     # Print comprehensive report
#     print("\n" + "=" * 50)
#     print("Evaluation Metrics:")
#     print(f"MSE:  {mse:.6f}")
#     print(f"MAE:  {mae:.6f}")
#     print(f"RMSE: {rmse:.6f}")
#     print(f"R²:   {r2:.6f}")
#     print(f"Directional Accuracy: {directional_acc:.2f}%")
#     print("=" * 50 + "\n")
#
#     print("Predictions vs Targets Statistics:")
#     stats = pd.DataFrame({
#         'Predictions': predictions,
#         'Targets': targets
#     }).describe()
#     print(stats)
#
#     print("\nFirst 20 Predictions vs Targets:")
#     for i in range(min(20, len(predictions))):
#         print(f"[{i}] Pred: {predictions[i]:.4f} | Target: {targets[i]:.4f} | "
#               f"Error: {predictions[i] - targets[i]:.4f} | "
#               f"Direction: {'✓' if np.sign(predictions[i]) == np.sign(targets[i]) else '✗'}")
#
#     return {
#         'mse': mse,
#         'mae': mae,
#         'rmse': rmse,
#         'r2': r2,
#         'directional_acc': directional_acc,
#         'predictions': predictions,
#         'targets': targets
#     }
def evaluate_model(model, X_test, y_test, seq_len, device, batch):
    """Evaluate the model on the test data using a sliding window of step 1"""
    Xs, ys = [], []
    for i in range(0, len(X_test) - seq_len + 1):
        Xs.append(X_test[i : i + seq_len])
        ys.append(y_test[i + seq_len - 1])
    X_seq = np.array(Xs)
    y_seq = np.array(ys)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.float32)
        ),
        batch_size=batch,
        shuffle=False
    )

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred_mean, _ = model(xb)
            preds.append(pred_mean.squeeze(-1).cpu().numpy())
            trues.append(yb.cpu().numpy())

    predictions = np.concatenate(preds)
    targets     = np.concatenate(trues)

    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    directional_acc = np.mean(np.sign(predictions) == np.sign(targets)) * 100
    r2 = r2_score(targets, predictions)

    print("\nEvaluation Metrics:")
    print(f"MSE:  {mse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²:   {r2:.6f}")
    print(f"Directional Accuracy: {directional_acc:.2f}%")

    stats = pd.DataFrame({'Predictions': predictions, 'Targets': targets}).describe()
    print("\nPrediction vs Target stats:\n", stats)

    print("\nFirst 20 comparisons:")
    for i in range(min(20, len(predictions))):
        print(f"[{i}] Pred: {predictions[i]:.4f}, Target: {targets[i]:.4f}, "
              f"Error: {predictions[i]-targets[i]:.4f}, "
              f"Dir: {'✓' if np.sign(predictions[i])==np.sign(targets[i]) else '✗'}")

    print(predictions)
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'directional_acc': directional_acc,
        'predictions': predictions,
        'targets': targets
    }

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seeds set to {seed} for reproducibility")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Financial Transformer for Time Series Prediction")
    parser.add_argument("--csv", required=True, help="CSV file with features")
    parser.add_argument("--seq", type=int, default=14, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seeds
    set_seed(args.seed)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"model_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    print(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    features = [col for col in df.columns if col not in ['Date', 'Close', 'PctChange']]
    print(f"Using {len(features)} features:", features)

    # Split data
    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))

    X_train = df[features].values[:train_size]
    y_train = df['PctChange'].values[:train_size]

    X_val = df[features].values[train_size:train_size + val_size]
    y_val = df['PctChange'].values[train_size:train_size + val_size]

    X_test = df[features].values[train_size + val_size:]
    y_test = df['PctChange'].values[train_size + val_size:]

    print(f"Data split: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test")

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Train model
    print("\nTraining model...")
    device = torch.device(args.device)
    best_model_state, best_val_loss, model = train_model(
        X_train, y_train, X_val, y_val,
        feat_dim=len(features),
        seq_len=args.seq,
        device=device,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        patience=args.patience
    )

    # Evaluate model
    print("\nEvaluating model...")
    results = evaluate_model(
        model, X_test, y_test, args.seq, device, args.batch
    )

    # Print results
    # print("\n=== Final Test Metrics ===")
    # print(f"MSE:  {results['mse']:.6f}")
    # print(f"MAE:  {results['mae']:.6f}")
    # print(f"RMSE: {results['rmse']:.6f}")
    # print(f"R²:   {results['r2']:.6f}")
    # print(f"Directional Accuracy: {results['directional_acc']:.2f}%")

    # Save model artifacts
    torch.save(model.state_dict(), output_dir / "model.pth")
    np.save(output_dir / "scaler_mean.npy", scaler.mean_)
    np.save(output_dir / "scaler_scale.npy", scaler.scale_)
    np.save(output_dir / "predictions.npy", results['predictions'])
    np.save(output_dir / "targets.npy", results['targets'])

    print(f"\nModel saved to {output_dir}")


if __name__ == "__main__":
    main()
