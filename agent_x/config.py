import os
import json
from pathlib import Path
from copy import deepcopy
from dotenv import load_dotenv

load_dotenv()

def load_config():
    DEFAULT_CFG = {
        "api_key": os.getenv("OKX_API_KEY", ""),
        "api_secret": os.getenv("OKX_API_SECRET", ""),
        "passphrase": os.getenv("OKX_PASSPHRASE", ""),
        "symbol": "XRP-USDT-SWAP",
        "timeframe": "3m",
        "tdMode": "isolated",
        "posMode": "long_short_mode",
        "data_source": "okx",
        "fees": 0.0005,
        "slippage": 0.0001,
        "history_bars": 5000,
        
        "walk_forward": {
            "train_bars": 3000,
            "test_bars": 2000,
            "pop": 30,
            "gens": 12,
            "topk": 6
        },
         
        "paths": {
            "state_dir": "state",
            "best_params": "state/best_params.json",
            "equity_curve": "state/equity_curve.csv",
            "trade_history": "state/trade_history.csv",
            "model_path": "state/lstm_model.keras"
        },
         
        "runtime": {
            "reoptimize_hours": 24,
            "ws_public": "wss://ws.okx.com:8443/ws/v5/public",
            "ws_private": "wss://ws.okx.com:8443/ws/v5/private",
            "base_url": "https://www.okx.com",
            "ws_public_demo": "wss://wspap.okx.com:8443/ws/v5/public",
            "ws_private_demo": "wss://wspap.okx.com:8443/ws/v5/private",
            "base_url_demo": "https://www.okx.com",
            "demo": True
        },
        
        "scaling": {
            "enabled": True,
            "thresholds": [0.01, 0.02],  # Increased thresholds for less frequent scaling to improve profitability
            "position_ratios": [1.0, 0.5],  # Adjusted ratios for better scaling
            "exit_strategy": "full"  # "partial" or "full" - partial closes positions in reverse order
        },
        
        "risk": {
            "max_open_orders": 2, # Maximum number of open orders allowed - Should this be 3 or 3.0?
            "leverage": 4, # Increased leverage for higher returns
            "backtest_equity": 30.0, # Used for backtesting and walk-forward analysis
            "max_daily_loss": 0.2, # Tighter daily loss limit
            "max_exposure": 0.8, # Lower exposure to control risk
            "risk_per_trade": 0.4 # Increased risk per trade for more position size
        },
        
        "lstm_params": {
            "sequence_length": 64,
            "epochs": 48,
            "batch_size": 48,
            "future_bars": 10
        },
        
        "lstm_strategy_params": {
            "indicator_weights": {
                "rsi": 0.4,
                "macd": 0.3,
                "bb": 0.3
            },
            "exit_mode": "intelligent",
            "atr_mult_sl": 3.1,
            "initial_trail_pct": 0.06,
            "profit_trigger_pct": 0.035,
            "tighter_trail_pct": 0.01,
            "lstm_disagreement_pct": 0.009,
            "prediction_threshold": 0.006
        }
    }

    cfg = deepcopy(DEFAULT_CFG)

    # Override with environment variables if set
    if os.getenv("OKX_SYMBOL"):
        cfg["symbol"] = os.getenv("OKX_SYMBOL")
    if os.getenv("OKX_TIMEFRAME"):
        cfg["timeframe"] = os.getenv("OKX_TIMEFRAME")
    if os.getenv("DATA_SOURCE"):
        cfg["data_source"] = os.getenv("DATA_SOURCE")
    if os.getenv("HISTORY_BARS"):
        cfg["history_bars"] = int(os.getenv("HISTORY_BARS"))
    if os.getenv("OKX_WS_PUBLIC_URL"):
        cfg["runtime"]["ws_public"] = os.getenv("OKX_WS_PUBLIC_URL")
    if os.getenv("OKX_WS_PRIVATE_URL"):
        cfg["runtime"]["ws_private"] = os.getenv("OKX_WS_PRIVATE_URL")
    if os.getenv("OKX_BASE_URL"):
        cfg["runtime"]["base_url"] = os.getenv("OKX_BASE_URL")
    if os.getenv("OKX_WS_PUBLIC_URL_DEMO"):
        cfg["runtime"]["ws_public_demo"] = os.getenv("OKX_WS_PUBLIC_URL_DEMO")
    if os.getenv("OKX_WS_PRIVATE_URL_DEMO"):
        cfg["runtime"]["ws_private_demo"] = os.getenv("OKX_WS_PRIVATE_URL_DEMO")
    if os.getenv("OKX_BASE_URL_DEMO"):
        cfg["runtime"]["base_url_demo"] = os.getenv("OKX_BASE_URL_DEMO")
    if os.getenv("OKX_DEMO"):
        cfg["runtime"]["demo"] = str(os.getenv("OKX_DEMO")).lower() in ("1", "true", "yes")

    # Remove credentials from config, rely on .env only
    cfg.pop("api_key", None)
    cfg.pop("api_secret", None)
    cfg.pop("passphrase", None)

    Path(cfg["paths"]["state_dir"]).mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(cfg["paths"]["state_dir"]) / "active_config.json"
    with snapshot_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    return cfg

def get_ultra_aggressive_params():
    """Ultra-aggressive parameters optimized for $25 starting capital"""
    from .strategy import LSTMStratParams

    return LSTMStratParams(
        prediction_threshold=0.0008,  # More aggressive threshold
        indicator_weights={'rsi': 0.45, 'macd': 0.35, 'bb': 0.2},  # Favor momentum more
        exit_mode='mechanical',
        atr_mult_sl=1.8,  # Even tighter stops
        initial_trail_pct=0.04,  # 4% trailing
        profit_trigger_pct=0.025,  # 2.5% profit target
        lstm_disagreement_pct=0.008  # Slightly less strict disagreement
    )

def load_config_for_small_cap(starting_capital: float = 25.0):
    """Load configuration optimized for small capital trading"""
    cfg = load_config()

    # Adjust risk parameters for small capital
    cfg["risk"]["backtest_equity"] = starting_capital
    cfg["risk"]["account_equity"] = starting_capital
    cfg["risk"]["risk_per_trade"] = 0.45  # Slightly higher risk per trade for small capital
    cfg["risk"]["max_daily_loss"] = 0.12  # Slightly looser daily loss limit

    # Adjust scaling thresholds for small capital
    cfg["scaling"]["thresholds"] = [0.008, 0.018]  # Lower thresholds for more frequent scaling
    cfg["scaling"]["position_ratios"] = [1.0, 0.6, 0.3]  # Adjusted position ratios for better scaling

    # Adjust LSTM strategy parameters for small capital
    from .strategy import LSTMStratParams
    cfg["lstm_params"] = LSTMStratParams(
        prediction_threshold=0.0015,  # More aggressive threshold
        indicator_weights={'rsi': 0.4, 'macd': 0.4, 'bb': 0.2},
        exit_mode='mechanical',
        atr_mult_sl=2.5,
        initial_trail_pct=0.045,
        profit_trigger_pct=0.055,
        lstm_disagreement_pct=0.006
    )

    return cfg

if __name__ == "__main__":
    cfg = load_config()
    print(json.dumps(cfg, indent=2))
