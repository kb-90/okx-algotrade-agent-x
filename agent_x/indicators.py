import pandas as pd, numpy as np

def ema(series, span): return series.ewm(span=span, adjust=False).mean()

def rsi(series, length=14):
    delta=series.diff()
    up=delta.clip(lower=0).rolling(length).mean()
    down=-delta.clip(upper=0).rolling(length).mean()
    rs=up/(down+1e-12)
    return 100-(100/(1+rs))

def atr(df, length=14):
    hl=df["high"]-df["low"]
    hc=(df["high"]-df["close"].shift()).abs()
    lc=(df["low"]-df["close"].shift()).abs()
    tr=pd.concat([hl,hc,lc],axis=1).max(axis=1)
    return tr.rolling(length).mean()

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def bollinger_bands(series, length=20, std_dev=2.0):
    sma = series.rolling(length).mean()
    std = series.rolling(length).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def adx(high, low, close, length=14):
    """Calculate Average Directional Index"""
    high_diff = high.diff()
    low_diff = low.diff()

    plus_dm = ((high_diff > low_diff) & (high_diff > 0)) * high_diff
    minus_dm = ((low_diff > high_diff) & (low_diff > 0)) * low_diff

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)

    atr_val = tr.rolling(length).mean()
    plus_di = 100 * (plus_dm.rolling(length).mean() / (atr_val + 1e-12))
    minus_di = 100 * (minus_dm.rolling(length).mean() / (atr_val + 1e-12))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12))
    adx_val = dx.rolling(length).mean()

    return adx_val

def detect_market_regime(df: pd.DataFrame) -> pd.Series:
    """Detect trending vs ranging market"""
    adx_val = adx(df['high'], df['low'], df['close'], length=14)
    regime = pd.Series(0, index=df.index, dtype=int)
    regime.loc[adx_val > 25] = 1   # Trending
    regime.loc[adx_val < 15] = -1  # Ranging
    return regime

def compute_features(df,p):
    x=df.copy()
    # Existing
    x["ema_fast"]=ema(x["close"],p.get("ema_fast", 12))
    x["ema_slow"]=ema(x["close"],p.get("ema_slow", 26))
    x["rsi"]=rsi(x["close"],p.get("rsi_len", 14))
    x["atr"]=atr(x,p.get("atr_len", 14)).bfill()
    x["trend"]=(x["ema_fast"]-x["ema_slow"])/(x["ema_slow"]+1e-12)
    x["vol_regime"]=(x["atr"]/x["close"]).rolling(p.get("regime_len", 48)).mean()

    # New Indicators
    macd_line, signal_line = macd(x["close"], p.get("macd_fast", 12), p.get("macd_slow", 26), p.get("macd_signal", 9))
    x["macd"] = macd_line
    x["macd_signal"] = signal_line

    upper_band, sma, lower_band = bollinger_bands(x["close"], p.get("bb_len", 20), p.get("bb_std", 2.0))
    x["bb_upper"] = upper_band
    x["bb_sma"] = sma
    x["bb_lower"] = lower_band
    x["bb_width"] = (upper_band - lower_band) / (sma + 1e-12)

    # Market regime detection
    x["regime"] = detect_market_regime(x)

    # Reorder columns
    feature_cols = [
        'close', 'open', 'high', 'low', 'volume', 'ema_fast', 'ema_slow',
        'rsi', 'atr', 'trend', 'vol_regime', 'macd', 'macd_signal',
        'bb_upper', 'bb_sma', 'bb_lower', 'bb_width', 'regime'
    ]
    x = x[feature_cols]

    # Fill NaN values with forward/backward fill to preserve more data
    x = x.bfill().ffill()

    # Only drop rows if all values are still NaN (shouldn't happen with proper data)
    return x.dropna(how='all')
