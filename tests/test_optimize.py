import pandas as pd
import numpy as np
from agent_x.optimize import EvoSearch
from agent_x.strategy import LSTMStratParams
from agent_x.indicators import compute_features

BTC_PRICE = 115000

def test_evo_search_runs():
    # 1. Setup
    length = 100
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

    cfg = {
        "fees": 0.001, 
        "slippage": 0.001, 
        "risk": {
            "backtest_equity": 10000, "vol_target": 0.1, "max_leverage": 2, 
            "max_daily_loss": 0.2, "max_exposure": 0.9, "risk_per_trade": 0.01
        },
        "walk_forward": {"pop": 2, "gens": 1, "topk": 1}
    }

    # Mock model for testing
    class MockModel:
        def predict_sequence(self, df):
            return np.array([0.02] * len(df))  # Mock bullish predictions
    
    model = MockModel()

    # Mock generate_signals to return tuple for scaling support
    def mock_generate_signals(df, current_positions=None):
        signals = pd.Series([1] * len(df), index=df.index)  # Mock main signals
        scaling_signals = [pd.Series([0] * len(df), index=df.index) for _ in range(2)]  # Mock scaling signals
        exit_signals = [pd.Series([0] * len(df), index=df.index) for _ in range(3)]  # Mock exit signals
        return signals, scaling_signals, exit_signals

    # Monkey patch the strategy's generate_signals method
    from agent_x.strategy import LSTMStrategy
    original_generate_signals = LSTMStrategy.generate_signals
    LSTMStrategy.generate_signals = mock_generate_signals

    try:
        # 2. Run EvoSearch
        evo = EvoSearch(cfg)
        best_params = evo.search(features, cfg, model)
    finally:
        # Restore original method
        LSTMStrategy.generate_signals = original_generate_signals

    # 3. Assertions
    assert isinstance(best_params, LSTMStratParams)
    assert best_params.prediction_threshold > 0
    assert best_params.atr_mult_sl > 0