from __future__ import annotations
import os
import time
import pandas as pd
import requests
from typing import Dict, List
from .utils import logger
from . import api_client

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

class DataInterface:
    def fetch_history(self, limit: int) -> pd.DataFrame:
        raise NotImplementedError("fetch_history must be implemented by subclasses")

class OKXData(DataInterface):
    def __init__(self, cfg: Dict, api_client: api_client.OKXAPIClient = None):
        self.cfg = cfg
        self.base_url = cfg["runtime"]["base_url"]
        self.symbol = cfg["symbol"]
        self.tf = cfg["timeframe"]
        self.api_client = api_client
        if self.api_client:
            self.market = self.api_client.market
            logger.info("OKXData is using authenticated API client.")
        else:
            self.market = None
            logger.warning("OKXData is using public REST endpoint. Data may be limited.")

    def fetch_history(self, limit: int) -> pd.DataFrame:
        logger.info(f"Fetching up to {limit} historical bars for {self.symbol} on {self.tf} timeframe...")
        all_data: List[pd.DataFrame] = []
        remaining = limit
        after_ts = None

        while remaining > 0:
            batch_size = min(100, remaining)
            data = None
            
            if self.market:
                resp = self.market.get_historical_candles(self.symbol, self.tf, batch_size, after_ts=after_ts)
                if resp and resp.get("code") == "0":
                    data = resp.get("data", [])
                else:
                    logger.warning(f"API client fetch failed: {resp.get('msg') if resp else 'No response'}. Retrying...")
                    time.sleep(2)
                    continue
            else:
                logger.error("Cannot fetch deep history without authenticated API client.")
                break

            if data:
                df = pd.DataFrame(data, columns=["ts", "o", "h", "l", "c", "vol", "volCcy", "volCcyQuote", "confirm"])
                df = df[["ts", "o", "h", "l", "c", "vol"]].astype(float)
                df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "vol": "volume"}, inplace=True)
                df.set_index("ts", inplace=True)
                
                all_data.append(df.iloc[::-1])
                after_ts = data[-1][0]
                remaining -= len(df)
                logger.info(f"Fetched batch of {len(df)} bars. {remaining} remaining.")
                if len(df) < batch_size:
                    logger.info("Reached end of available history for this batch.")
                    break
            else:
                logger.warning("No data received in this batch. Stopping fetch.")
                break
            
            time.sleep(0.5)
        
        if not all_data:
            raise RuntimeError("Failed to fetch any historical data from OKX.")

        full_df = pd.concat(all_data).sort_index()
        full_df = full_df[~full_df.index.duplicated(keep="first")]
        if len(full_df) > limit:
            full_df = full_df.tail(limit)
        logger.info(f"Successfully fetched exactly {len(full_df)} unique bars.")
        return full_df

class CCXTData(DataInterface):
    def __init__(self, cfg: Dict):
        if not CCXT_AVAILABLE:
            raise ImportError("ccxt library is not installed. Please install it to use CCXTData.")
        self.cfg = cfg
        self.symbol = cfg["symbol"]
        self.tf = cfg["timeframe"]
        self.exchange = ccxt.okx({
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'},
        })
        if cfg["runtime"].get("demo", True):
            self.exchange.set_sandbox_mode(True)
        logger.info("CCXTData initialized with OKX exchange in sandbox mode: " + str(cfg["runtime"].get("demo", True)))

    def fetch_history(self, limit: int) -> pd.DataFrame:
        logger.info(f"Fetching up to {limit} historical bars for {self.symbol} on {self.tf} timeframe using CCXT...")
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df.set_index("ts", inplace=True)
            logger.info(f"Successfully fetched {len(df)} bars using CCXT.")
            return df
        except Exception as e:
            logger.error(f"CCXT fetch_history failed: {e}")
            raise
