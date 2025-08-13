import yfinance as yf
import pandas as pd

def download_data(tickers, start='2010-01-01', end='2023-12-31'):
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)
    all_data = {}
    for ticker in tickers:
        df = data[ticker].copy()
        df['Ticker'] = ticker
        df = df[['Ticker', 'Close']].dropna()
        all_data[ticker] = df
    return all_data

def save_data(all_data, path='../data/'):
    for ticker, df in all_data.items():
        df.to_csv(f"{path}{ticker}.csv", index=True)
