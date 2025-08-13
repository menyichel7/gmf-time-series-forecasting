import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(ts, look_back=30):
    scaler = MinMaxScaler()
    ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(ts_scaled) - look_back):
        X.append(ts_scaled[i:i + look_back])
        y.append(ts_scaled[i + look_back])
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model
