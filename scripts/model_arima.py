import pmdarima as pm

def train_arima(ts):
    model = pm.auto_arima(ts, seasonal=False, stepwise=True, suppress_warnings=True)
    return model

def forecast_arima(model, periods=30):
    forecast = model.predict(n_periods=periods)
    return forecast
