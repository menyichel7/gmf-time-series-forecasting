import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def plot_rolling_stats(ts, window=30):
    rolmean = ts.rolling(window).mean()
    rolstd = ts.rolling(window).std()
    plt.figure(figsize=(10, 4))
    plt.plot(ts, label='Original')
    plt.plot(rolmean, label='Rolling Mean')
    plt.plot(rolstd, label='Rolling Std')
    plt.legend()
    plt.title('Rolling Mean & Standard Deviation')
    plt.tight_layout()
    plt.show()

def test_stationarity(ts):
    result = adfuller(ts.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    return result[1] < 0.05
