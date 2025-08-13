import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

def optimize_portfolio(price_df):
    mu = mean_historical_return(price_df)
    S = CovarianceShrinkage(price_df).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    return cleaned_weights
