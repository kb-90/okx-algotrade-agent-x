import os
import numpy as np
import pandas as pd
import joblib
import json
from .utils import logger

# Custom Scaler to remove sklearn dependency
class SimpleScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit_transform(self, data):
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        # Add a small epsilon to avoid division by zero if a column is constant
        return (data - self.min) / (self.max - self.min + 1e-8)

    def transform(self, data):
        if self.min is None or self.max is None:
            raise RuntimeError("Scaler has not been fit yet.")
        return (data - self.min) / (self.max - self.min + 1e-8)

    def inverse_transform(self, data):
        if self.min is None or self.max is None:
            raise RuntimeError("Scaler has not been fit yet.")
        return data * (self.max - self.min + 1e-8) + self.min

    @property
    def n_features_in_(self):
        return len(self.min) if self.min is not None else 0

def create_sequences(data, sequence_length):
    X = []
    for i in range(len(data) - sequence_length + 1):
        X.append(data[i:(i + sequence_length), :])
    return np.array(X)

def create_training_dataset(data, sequence_length, future_bars=3):
    X, y = [], []
    for i in range(len(data) - sequence_length - future_bars + 1):
        X.append(data[i:(i + sequence_length), :])
        y.append(data[i + sequence_length + future_bars - 1, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

class LSTMModel:
    def __init__(self, sequence_length=60, epochs=10, batch_size=32, future_bars=7):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.future_bars = future_bars
        self.scaler = SimpleScaler() # Use the custom scaler
        self.model = None
        self.feature_columns = None

    def train(self, df_features: pd.DataFrame):
        self.feature_columns = df_features.columns.tolist()
        scaled_data = self.scaler.fit_transform(df_features.values)
        X_train, y_train = create_training_dataset(scaled_data, self.sequence_length, self.future_bars)
        if X_train.shape[0] == 0:
            raise ValueError("Not enough data to create a single training sequence.")
        self.model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

    def predict(self, df_features: pd.DataFrame) -> float:
        if self.model is None: raise RuntimeError("Model not trained.")
        if self.feature_columns is None: raise RuntimeError("Model is trained but feature_columns is not set.")
        
        df_aligned = df_features[self.feature_columns].iloc[-self.sequence_length:]
        scaled_sequence = self.scaler.transform(df_aligned.values)
        X_pred = np.array([scaled_sequence])
        predicted_price_scaled = self.model.predict(X_pred)
        dummy_array = np.zeros((1, self.scaler.n_features_in_))
        dummy_array[0, 0] = predicted_price_scaled[0, 0]
        predicted_price = self.scaler.inverse_transform(dummy_array)[0, 0]
        return predicted_price

    def predict_sequence(self, df_features: pd.DataFrame) -> np.ndarray:
        if self.model is None: raise RuntimeError("Model not trained.")
        if self.feature_columns is None: raise RuntimeError("Model is trained but feature_columns is not set.")

        df_aligned = df_features[self.feature_columns]
        scaled_data = self.scaler.transform(df_aligned.values)

        if np.any(np.isnan(scaled_data)) or np.any(np.isinf(scaled_data)):
            logger.warning("Scaled data contains NaN or Inf values. Skipping prediction.")
            return np.array([])
        sequences = create_sequences(scaled_data, self.sequence_length)
        if len(sequences) == 0: return np.array([])
        predicted_scaled = self.model.predict(sequences)
        dummy_array = np.zeros((len(predicted_scaled), self.scaler.n_features_in_))
        dummy_array[:, 0] = predicted_scaled.flatten()
        predictions = self.scaler.inverse_transform(dummy_array)[:, 0]
        return predictions

    def save(self, model_path: str):
        if self.model is None: raise RuntimeError("No model to save.")
        self.model.save(model_path)
        scaler_path = os.path.splitext(model_path)[0] + '_scaler.gz'
        joblib.dump(self.scaler, scaler_path)

        columns_path = os.path.splitext(model_path)[0] + '_columns.json'
        with open(columns_path, 'w') as f:
            json.dump(self.feature_columns, f)

    def load(self, model_path: str):
        from tensorflow.keras.models import load_model
        self.model = load_model(model_path)
        scaler_path = os.path.splitext(model_path)[0] + '_scaler.gz'
        self.scaler = joblib.load(scaler_path)

        columns_path = os.path.splitext(model_path)[0] + '_columns.json'
        try:
            with open(columns_path, 'r') as f:
                self.feature_columns = json.load(f)
        except FileNotFoundError:
            self.feature_columns = None
            logger.warning("Feature columns file not found. Model may not work correctly if feature set has changed.")
