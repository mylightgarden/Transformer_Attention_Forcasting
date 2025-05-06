# Transformer_Attention_Forcasting
Transformers are highly effective at capturing temporal dependencies and complex patterns in sequences, making them suitable for modeling financial markets, which are inherently sequential and noisy.

This project aims to predict daily stock price movements by leveraging advanced deep learning models trained on historical SPY data enriched with technical indicators. To ensure stable and unbiased training, the data was normalized using a StandardScaler, which standardizes features by removing the mean and scaling to unit variance.

Since Transformers are capable of capturing complex patterns, the following financial indicators were calculated and then worked as features that were fed as input data:
• Price-derived features:
High_to_Close, Low_to_Close,
Open_to_Close, Overnight_Gap
• Volume feature: Volume_z (standardized)
• Trend and momentum: MACD, RSI, Stochastic
RSI (K/D)
• Volatility: ATR, Bollinger Bands, rolling std,
moving_avg_5
• Cumulative flow: OBV
• Returns: log_returns, ROC
The core architecture features a custom-built Transformer encoder, which processes sequences of past market data enriched with positional encoding. The output is a single scalar prediction.

Instead of standard Transformer layers, custom-designed layers were developed to better align with the characteristics of stock data. For example, attention bias was disabled to reduce overfitting, and GELU was used instead of ReLU as the activation function to better handle volatile regimes.

The training pipeline included creating sliding windows of data for both training and validation, applying early stopping to prevent overfitting, and evaluating the model on unseen test data for generalization. AdamW is used as the optimizer.

A customed loss function is intruduced:
Since the stock market tends to increase in a roughly linear fashion over the long term, the model could exploit this trend by finding a few ”sweet spots” and predicting similar positive values each day. This issue did occur: when using a simple mean squared error (MSE) as the loss function, the model initially produced a nearly constant positive output. To address this, a directional penalty is introduced into the loss function:
sign penalty = ReLU(−sign(pred) · target).mean()

Four key metrics commonly used to evaluate financial stock prediction models are MSE, MAE, RMSE, and directional accuracy, which respectively measure error magnitude, average deviation, interpretability, and trend prediction accuracy.

A weight analysis was performed on the transformer's input projection layer to identify which features it relies on most. By averaging the absolute weights across embedding dimensions, I found that the model heavily emphasizes the K–D difference, day-of-week, and 5-day volatility, indicating a preference for short-term momentum and weekly patterns. In contrast, raw volume and its z-score were least influential, suggesting volume has a minor role in the model’s decision-making.
