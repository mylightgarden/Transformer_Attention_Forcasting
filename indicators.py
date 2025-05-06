'''
Author: Sophie Zhao
'''

import pandas as pd
import numpy as np


def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = data['Close'].diff()

    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    gain_ewm = gain.ewm(alpha=1 / period, adjust=False).mean()
    loss_ewm = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = gain_ewm / loss_ewm

    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi

    return data


def calculate_stoch_rsi(data: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
    data = calculate_rsi(data, period)

    epsilon = 1e-10  # To prevent division by zero
    rsi_min = data['RSI'].rolling(window=period, min_periods=period).min()
    rsi_max = data['RSI'].rolling(window=period, min_periods=period).max()

    stoch_rsi = (data['RSI'] - rsi_min) / (rsi_max - rsi_min + epsilon)
    data['StochRSI'] = stoch_rsi

    data['K'] = data['StochRSI'].rolling(window=smooth_k, min_periods=1).mean() * 100
    data['D'] = data['K'].rolling(window=smooth_d, min_periods=1).mean()

    return data


def calculate_roc(data, periods=10):
    roc = ((data['Close'] - data['Close'].shift(periods)) / data['Close'].shift(periods)) * 100
    data['ROC'] = roc
    return data


def calculate_bollinger_bands(data, window=10, num_of_std=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()

    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)

    data['B_percent'] = (data['Close'] - lower_band) / (upper_band - lower_band)

    return data


def calculate_ATR(data, window=14):
    data['High_Low'] = data['High'] - data['Low']
    data['High_Close'] = np.abs(data['High'] - data['Close'].shift())
    data['Low_Close'] = np.abs(data['Low'] - data['Close'].shift())

    data['True_Range'] = data[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
    data['ATR'] = data['True_Range'].rolling(window=window).mean()

    data.drop(columns=['High_Low', 'High_Close', 'Low_Close', 'True_Range'], inplace=True)

    return data


def calculate_MACD(data, span_short=12, span_long=26, span_signal=9):
    ema_short = data['Close'].ewm(span=span_short, adjust=False).mean()
    ema_long = data['Close'].ewm(span=span_long, adjust=False).mean()

    data['MACD'] = ema_short - ema_long
    data['MACD_Signal'] = data['MACD'].ewm(span=span_signal, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    return data


def calculate_CMF(data, window=20):
    mf_multiplier = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    mf_volume = mf_multiplier * data['Volume']

    data['CMF'] = mf_volume.rolling(window=window).sum() / data['Volume'].rolling(window=window).sum()

    return data


def calculate_OBV(data, window=20):
    sign = np.sign(data['Close'].diff()).fillna(0)
    data['OBV'] = (sign * data['Volume']).cumsum()

    # rolling mean & std of OBV
    rol_mean = data['OBV'].rolling(window).mean()
    rol_std = data['OBV'].rolling(window).std()

    # z‑score over past 20 days
    data['OBV_z'] = (data['OBV'] - rol_mean) / rol_std

    # drop the first 'window' rows which will be NaN
    data = data.dropna(subset=['OBV_z'])

    return data
