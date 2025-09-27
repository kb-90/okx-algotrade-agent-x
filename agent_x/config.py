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
        "timeframe": "15m",
        "tdMode": "isolated",
        "posMode": "long_short_mode",
        "data_source": "okx",
        "fees": 0.0005,
        "slippage": 0.00015,
        "history_bars": 4000,
        
        "walk_forward": {
            "train_bars": 2750,
            "test_bars": 1250,
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
        
        "risk": {
            "max_open_orders": 2, # Maximum number of open orders allowed
            "leverage": 5, # Increased leverage for higher returns
            "backtest_equity": 30.0, # Used for backtesting and walk-forward analysis
            "max_daily_loss": 0.15, # Tighter daily loss limit
            "max_exposure": 0.8, # Lower exposure to control risk
            "risk_per_trade": 0.4, # Before leverage, portion of equity risked per trade
            "position_ratios": [1.0, 0.5]  # Configurable position size ratios for scaling (initial, second position)
        },
        
        "lstm_params": {
            "sequence_length": 100,
            "epochs": 50,
            "batch_size": 50,
            "future_bars": 16
        },
        
        "lstm_strategy_params": {
            "exit_mode": "intelligent",
            "atr_mult_sl": 2.3,
            "initial_trail_pct": 0.06,
            "profit_trigger_pct": 0.06,
            "tighter_trail_pct": 0.01,
            "lstm_disagreement_pct": 0.005,
            "prediction_threshold": 0.0015,
            "indicator_weights": {
                "rsi": 0.3,
                "macd": 0.4,
                "bb": 0.3
            }  
        }
        # possible additions to consider (contemplating improvements to strategy):
            # "min_trade_duration": 1,
            # "min_confidence": 0.5
            # "max_holding_period": 24,
            # "take_profit_multiplier": 2.0,
    }

    cfg = deepcopy(DEFAULT_CFG)

    # Override with environment variables if set
    if os.getenv("OKX_SYMBOL"):
        cfg["symbol"] = os.getenv("OKX_SYMBOL")
    if os.getenv("OKX_TIMEFRAME"):
        cfg["timeframe"] = os.getenv("OKX_TIMEFRAME")
    if os.getenv("DATA_SOURCE"):
        cfg["data_source"] = os.getenv("DATA_SOURCE")
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

if __name__ == "__main__":
    cfg = load_config()
    print(json.dumps(cfg, indent=2))
