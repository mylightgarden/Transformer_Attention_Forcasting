'''
Author: Sophie Zhao
'''

import yfinance as yf
from indicators import calculate_rsi, calculate_roc, calculate_stoch_rsi, calculate_bollinger_bands, calculate_ATR, \
    calculate_MACD, calculate_OBV
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_data(ticker='NVDA', if_PctChange=False, if_indicators=False):
    ticker = yf.Ticker(ticker)
    df = ticker.history(start="2003-12-04", interval="1d").reset_index()
    # print(df.head())
    # print(df.shape)

    df['High_to_Close'] = 100 * (df['High'] - df['Close']) / df['Close']
    df['Low_to_Close'] = 100 * (df['Low'] - df['Close']) / df['Close']
    df['Open_to_Close'] = 100 * (df['Open'] - df['Close']) / df['Close']
    df['Overnight_Gap'] = 100 * (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

    scaler = StandardScaler()
    df['Volume_z'] = scaler.fit_transform(df[['Volume']])

    if if_indicators:
        df = calculate_ATR(df)
        df = calculate_MACD(df)
        df = calculate_rsi(df)
        df = calculate_stoch_rsi(df)
        df['K_D_diff'] = df['K'] - df['D']
        df = calculate_bollinger_bands(df)
        df = calculate_roc(df)
        df = calculate_OBV(df, window=20)
        df['log_returns'] = np.log(df['Close']).diff()
        df['volatility'] = df['log_returns'].rolling(5).std()
        df['moving_avg_5'] = df['Close'].rolling(5).mean()

    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month

    if if_PctChange:
        df['PctChange'] = 100 * df['Close'].pct_change().shift(-1)
        # Drop last row
        df = df[:-1]

    df = df.drop(columns=['Open', 'High', 'Low', 'OBV', 'Capital Gains',],
                 errors='ignore')

    return df.dropna()


# df_qqq = get_data('QQQ', True, True)
# # print(df_qqq.to_string())
# df_smh = get_data('SMH', True, True)
# df_nvda = get_data('NVDA', True, True)
# df_msft = get_data('MSFT', True, True)
# df_spy = get_data('SPY', True, True)
df_ko = get_data('KO', True, True)

# df_qqq.to_csv('qqq_features.csv', index=False)
# df_smh.to_csv('smh_features.csv', index=False)
# df_nvda.to_csv('nvda_features.csv', index=False)
# df_msft.to_csv('msft_features.csv', index=False)
# df_spy.to_csv('spy_features.csv', index=False)
df_ko.to_csv('ko_features.csv', index=False)
