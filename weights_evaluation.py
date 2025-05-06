'''
Author: Sophie Zhao
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from stock_pct_change_transformer import SimpleTransformer

# Load the trained model
device = torch.device("cpu")
model = SimpleTransformer(
    feat_dim=25,
    seq_len=14,
    d_model=64,
    nhead=4,
    num_layers=2,
    dropout=0.1
).to(device)
state = torch.load("./output/model_20250425_163843/model.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()

# Extract the weight matrix: shape (d_model, feat_dim)
W = model.input_proj.weight.data.cpu().numpy()

# Compute per-feature scores: mean absolute weight across all d_model neurons
feat_scores = np.mean(np.abs(W), axis=0)

# Plot
feature_names = ['Volume', 'Dividends', 'Stock Splits', 'High_to_Close', 'Low_to_Close', 'Open_to_Close',
                 'Overnight_Gap', 'Volume_z', 'ATR', 'MACD', 'MACD_Signal', 'MACD_Hist', 'RSI', 'StochRSI', 'K', 'D',
                 'K_D_diff', 'B_percent', 'ROC', 'OBV_z', 'log_returns', 'volatility', 'moving_avg_5', 'DayOfWeek',
                 'Month']

inds = np.argsort(feat_scores)[::-1]

plt.figure()
plt.bar(np.array(feature_names)[inds], feat_scores[inds])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Mean |weight|")
plt.title("Feature importance via input_proj.weight (Transformer)")
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150)
print("Saved plot to feature_importances.png")
