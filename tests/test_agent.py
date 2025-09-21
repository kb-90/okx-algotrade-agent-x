import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from dataclasses import asdict
from agent_x.agent import Agent
from agent_x.strategy import LSTMStratParams

BTC_PRICE = 115000

# A dummy LSTM model that bypasses actual training and prediction
class DummyLSTMModel:
    def __init__(self, **kwargs):
        self.sequence_length = 60

    def train(self, df_features):
        pass

    def save(self, path):
        with open(path, 'w') as f:
            f.write("dummy model")

    def load(self, path):
        pass

    def predict_sequence(self, df_features):
        return df_features['close'].values[self.sequence_length-1:] * 1.02

def test_agent_run_once(monkeypatch, tmp_path):
    # 1. Setup a dummy config
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    cfg = {
        "symbol": "BTC-USDT-SWAP", "timeframe": "1h", "history_bars": 200,
        "fees": 0.001, "slippage": 0.001,
        "risk": {
            "vol_target": 0.1, "max_leverage": 2,
            "max_daily_loss": 0.2, "max_exposure": 0.9, "risk_per_trade": 0.1,
            "backtest_equity": 50
        },
        "walk_forward": {"train_bars": 150, "pop": 2, "gens": 1, "topk": 1},
        "runtime": {"reoptimize_hours": 0}, # Re-optimize every time for test
        "order": {"tdMode": "cross"},
        "paths": {
            "state_dir": str(state_dir),
            "best_params": str(state_dir / "best_params.json"),
            "equity_curve": str(state_dir / "equity_curve.csv"),
            "model_path": str(state_dir / "lstm_model.h5")
        }
    }

    # 2. Mock external dependencies
    class DummyData:
        def __init__(self, cfg, api_client=None):
            pass
        def fetch_history(self, _):
            length = 200
            idx = pd.date_range("2023-01-01", periods=length, freq="h")
            base_price = np.arange(length) * 10 + BTC_PRICE
            return pd.DataFrame({
                'open': base_price - np.random.rand(length) * 50,
                'high': base_price + np.random.rand(length) * 50 + 25,
                'low': base_price - np.random.rand(length) * 50 - 25,
                'close': base_price,
                'volume': np.random.rand(length) * 100 + 10
            }, index=idx)

    monkeypatch.setattr("agent_x.agent.OKXData", DummyData)
    monkeypatch.setattr("agent_x.agent.LSTMModel", DummyLSTMModel)
    monkeypatch.setattr("agent_x.agent.OKXTrader", MagicMock)

    # Mock generate_signals to return tuple for scaling support
    def mock_generate_signals(self, df, current_positions=None):
        signals = pd.Series([1] * len(df), index=df.index)  # Mock main signals
        scaling_signals = [pd.Series([0] * len(df), index=df.index) for _ in range(2)]  # Mock scaling signals
        exit_signals = [pd.Series([0] * len(df), index=df.index) for _ in range(3)]  # Mock exit signals
        return signals, scaling_signals, exit_signals

    monkeypatch.setattr("agent_x.strategy.LSTMStrategy.generate_signals", mock_generate_signals)

    # 3. Run the agent
    agent = Agent(cfg)
    metrics = agent.run_once()

    # 4. Assertions
    assert "Sharpe" in metrics
    assert metrics["Sharpe"] is not None
    assert os.path.exists(cfg["paths"]["best_params"])
    assert os.path.exists(cfg["paths"]["equity_curve"])
    assert os.path.exists(cfg["paths"]["model_path"])
