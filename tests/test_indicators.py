import pandas as pd
import numpy as np
from agent_x import indicators

def make_series(length=50):
    # Use a longer series to ensure indicators have valid values after warm-up
    return pd.Series(np.random.rand(length) * 10 + 50)

def make_df(length=50):
    return pd.DataFrame({
        "open": np.random.rand(length) * 10 + 50,
        "high": np.random.rand(length) * 10 + 55,
        "low": np.random.rand(length) * 10 + 45,
        "close": np.random.rand(length) * 10 + 50,
        "volume": np.random.rand(length) * 1000
    })

def test_ema_and_rsi():
    s = make_series()
    e = indicators.ema(s, span=5)
    r = indicators.rsi(s, length=14)
    assert not e.isnull().any()
    # RSI will have NaNs at the start, so check after the lookback period
    assert not r.iloc[14:].isnull().any()
    assert r.max() <= 100 and r.min() >= 0

def test_atr():
    df = make_df()
    a = indicators.atr(df, length=14)
    assert len(a) == len(df)
    assert not a.iloc[14:].isnull().any()

def test_macd():
    s = make_series()
    macd_line, signal_line = indicators.macd(s)
    # MACD has a long warm-up period (slow EMA + signal line), check after it
    assert not macd_line.iloc[26:].isnull().any()
    assert not signal_line.iloc[35:].isnull().any()

def test_bollinger_bands():
    s = make_series()
    upper, sma, lower = indicators.bollinger_bands(s, length=20)
    # Check for NaNs after the lookback period
    assert not upper.iloc[20:].isnull().any()
    assert not sma.iloc[20:].isnull().any()
    assert not lower.iloc[20:].isnull().any()
    # Check that bands are correctly ordered
    assert (upper.iloc[20:] >= sma.iloc[20:]).all()
    assert (lower.iloc[20:] <= sma.iloc[20:]).all()

def test_compute_features():
    df = make_df(100)
    params = {
        "ema_fast": 12, "ema_slow": 26, "rsi_len": 14, "atr_len": 14, 
        "regime_len": 48, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "bb_len": 20, "bb_std": 2.0
    }
    features = indicators.compute_features(df, params)
    assert not features.isnull().values.any()
    expected_cols = ['macd', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_width']
    for col in expected_cols:
        assert col in features.columns