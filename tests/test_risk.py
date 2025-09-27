from datetime import datetime
import pandas as pd
from agent_x.risk import RiskConfig, RiskManager

def test_position_and_circuit():
    cfg = RiskConfig(backtest_equity=50, leverage=2, max_daily_loss=0.1, max_exposure=0.9, risk_per_trade=0.1)
    r = RiskManager(cfg, lot_size=0.1)

    # Test position sizing for initial position
    # Create a DataFrame with enough rows for ATR calculation
    df = pd.DataFrame({
        'close': [100] * 20,
        'high': [101] * 20,
        'low': [99] * 20,
        'atr': [1.0] * 20  # Add ATR column for risk calculation
    })
    sz = r.get_position_size(df, 1, 100, position_level=0)
    assert sz > 0

    # Test position sizing for scaling positions
    sz_scale1 = r.get_position_size(df, 1, 100, position_level=1)
    sz_scale2 = r.get_position_size(df, 1, 100, position_level=2)
    print(f"Base size: {sz}, Scale1: {sz_scale1}, Scale2: {sz_scale2}")  # Debug print
    assert sz_scale1 < sz  # Scaling positions should be smaller
    assert sz_scale2 < sz_scale1  # Each scaling level should be smaller

    # Test with no ATR (should return 0 due to no ATR available)
    df_no_atr = pd.DataFrame({'close': [100]})
    sz_no_atr = r.get_position_size(df_no_atr, 1, 100, position_level=0)
    assert sz_no_atr == 0  # Should return 0 when no ATR is available

# - Additional tests can be added here for other indicators as needed.
