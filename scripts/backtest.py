import pandas as pd

def calculate_returns(price_df):
    return price_df.pct_change().dropna()

def backtest_strategy(returns_df, weights):
    weighted_returns = returns_df @ pd.Series(weights)
    cumulative_returns = (1 + weighted_returns).cumprod()
    return cumulative_returns
