import os
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Optional, Callable

logging.basicConfig(level=logging.INFO, format="{asctime} | {levelname} | {name} | {message}", style='{')
logger = logging.getLogger("agent_x")

class TradingError(Exception):
    """Custom exception for trading-related errors"""
    pass

def safe_api_call(func: Callable, *args, **kwargs) -> Any:
    """Wrapper for API calls with consistent error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"API call failed: {e}")
        raise TradingError(f"API operation failed: {e}") from e

def to_ts_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def from_ts_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc)

def safe_read_json(path, default):
    try:
        with open(path,"r") as f: return json.load(f)
    except: return default

def safe_write_json(path, obj):
    tmp = path+".tmp"
    with open(tmp,"w") as f: json.dump(obj,f,indent=2)
    os.replace(tmp,path)
    
def clean_old_params(params_path: str) -> None:
    """Remove incompatible parameter files."""
    path = Path(params_path)
    if path.exists():
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Check if it has old structure
            if 'composite_confirm' in data or 'trend_weight' in data:
                logger.info(f"Removing old parameter file: {path}")
                path.unlink()  # Delete the file
                logger.info("Old parameters removed. New optimization will create fresh parameters.")
        except Exception as e:
            logger.warning(f"Could not check/clean old params: {e}")