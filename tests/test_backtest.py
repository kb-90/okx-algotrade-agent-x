import pandas as pd
import numpy as np
from agent_x.strategy import LSTMStratParams, LSTMStrategy
from agent_x.risk import RiskConfig, RiskManager
from agent_x.backtest import Backtester
from agent_x.indicators import compute_features

BTC_PRICE = 115000

def make_test_data(length=100):
    """Creates a dataframe with features and predictions for testing."""
    idx = pd.date_range("2023-01-01", periods=length, freq="h")
    base_price = np.arange(length) * 10 + BTC_PRICE
    df = pd.DataFrame({
        'open': base_price - np.random.rand(length) * 50,
        'high': base_price + np.random.rand(length) * 50 + 25,
        'low': base_price - np.random.rand(length) * 50 - 25,
        'close': base_price,
        'volume': np.random.rand(length) * 100 + 10
    }, index=idx)
    
    features = compute_features(df, LSTMStratParams().__dict__)
    predictions = pd.Series(features['close'] * 1.02, index=features.index)
    
    return features, predictions

def test_backtest_run_and_metrics():
    # 1. Setup
    features, predictions = make_test_data()
    params = LSTMStratParams(prediction_threshold=0.01)
    risk_cfg = RiskConfig(backtest_equity=10000, vol_target=0.15, max_leverage=3, max_daily_loss=0.1, max_exposure=0.9, risk_per_trade=0.1)
    risk = RiskManager(risk_cfg, lot_size=0.1)
    
    # Mock model for testing
    class MockModel:
        def predict_sequence(self, df):
            return np.array([0.02] * len(df))  # Mock bullish predictions

    model = MockModel()
    strat = LSTMStrategy(params, risk, model)

    # Mock generate_signals to return tuple for scaling support
    def mock_generate_signals(df, current_positions=None):
        signals = pd.Series([1] * len(df), index=df.index)  # Mock main signals
        scaling_signals = [pd.Series([0] * len(df), index=df.index) for _ in range(2)]  # Mock scaling signals
        exit_signals = [pd.Series([0] * len(df), index=df.index) for _ in range(3)]  # Mock exit signals
        return signals, scaling_signals, exit_signals

    strat.generate_signals = mock_generate_signals
    
    cfg = {
        "fees": 0.001,
        "slippage": 0.001,
        "risk": risk_cfg.__dict__
    }
    bt = Backtester(cfg, strat)

    # 2. Run backtest
    metrics = bt.run(features)

    # 3. Assertions
    assert isinstance(metrics, dict)
    assert "Sharpe" in metrics
    assert metrics["Sharpe"] is not None
