from __future__ import annotations
import os
import json
import queue
import time
import math
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict
from pathlib import Path

import pandas as pd
import numpy as np

from .utils import safe_read_json, safe_write_json, logger
from .data import OKXData, CCXTData, DataInterface
from .indicators import compute_features
from .model import LSTMModel
from .strategy import LSTMStratParams, LSTMStrategy
from .risk import RiskManager, RiskConfig
from .backtest import Backtester
from .optimize import EvoSearch
from .trader import OKXTrader

try:
    from . import api_client
    API_CLIENT_AVAILABLE = True
except ImportError:
    API_CLIENT_AVAILABLE = False

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False


class Agent:
    def __init__(self, cfg: Dict, api_client: api_client.OKXAPIClient = None):

        self.cfg = cfg
        self.api_client = api_client
        self.data = self._create_data_interface(cfg, api_client)

        self.model = LSTMModel(**cfg.get("lstm_params", {}))
        self.params = LSTMStratParams()
        self.trader = OKXTrader(cfg, api_client, self.params)

        # Ensure model path uses modern .keras extension
        self.cfg["paths"]["model_path"] = os.path.join(self.cfg["paths"]["state_dir"], "lstm_model.keras")

        # Load best params if present
        bp_path = self.cfg["paths"]["best_params"]
        loaded = safe_read_json(bp_path, None)
        if loaded:
            try:
                self.params = LSTMStratParams(**loaded)
                logger.info(f"Loaded best params from disk: {self.params}")
            except TypeError as e:
                logger.warning(f"Saved parameters don't match current structure: {e}")
                logger.info("Converting old parameter structure or using defaults...")
                try:
                    converted_params = self._convert_old_params(loaded)
                    self.params = LSTMStratParams(**converted_params)
                    logger.info(f"Converted old params: {self.params}")
                except Exception as conv_e:
                    logger.warning(f"Could not convert old params: {conv_e}")
                    logger.info("Using default parameters")
                    self.params = LSTMStratParams()

        self._last_reopt = None
        self.entry_px = np.nan
        self.highest_profit = 0.0
        self.position_levels = 0
        self.current_positions = []
        self.last_scale_time = None

    def _create_data_interface(self, cfg: Dict, api_client: api_client.OKXAPIClient = None) -> DataInterface:
        """Create the appropriate data interface based on configuration."""
        is_demo = cfg.get("runtime", {}).get("demo", True)
        data_source = cfg.get("data_source", "okx").lower()

        if is_demo:
            # Demo mode: use OKX REST API
            if data_source == "ccxt":
                logger.info("Using CCXT data interface for demo market data.")
                return CCXTData(cfg)
            else:
                logger.info("Using OKX data interface for demo market data.")
                return OKXData(cfg, api_client)
        else:
            # Live mode: use CCXT
            logger.info("Using CCXT data interface for live market data.")
            return CCXTData(cfg)

    def _convert_old_params(self, old_params: dict) -> dict:
        new_params = {}
        new_params['prediction_threshold'] = old_params.get('prediction_threshold', 0.01)

        if 'trend_weight' in old_params:
            new_params['indicator_weights'] = {
                'rsi': old_params.get('rsi_weight', 0.33),
                'macd': old_params.get('macd_weight', 0.33),
                'bb': old_params.get('bb_weight', 0.34)
            }
        elif 'indicator_weights' in old_params:
            new_params['indicator_weights'] = old_params['indicator_weights']
        else:
            new_params['indicator_weights'] = {'rsi': 0.33, 'macd': 0.33, 'bb': 0.34}

        new_params['exit_mode'] = old_params.get('exit_mode', 'intelligent')
        new_params['atr_mult_sl'] = old_params.get('atr_mult_sl', 2.0)
        new_params['initial_trail_pct'] = old_params.get('initial_trail_pct', 0.05)
        new_params['profit_trigger_pct'] = old_params.get('profit_trigger_pct', 0.1)
        new_params['tighter_trail_pct'] = old_params.get('tighter_trail_pct', 0.02)
        new_params['lstm_disagreement_pct'] = old_params.get('lstm_disagreement_pct', 0.03)
        return new_params

    def walk_forward_optimize(self, hist: pd.DataFrame) -> LSTMStratParams:
        wf = self.cfg["walk_forward"]
        train_data = hist.tail(wf["train_bars"])
        logger.info(f"Using {len(train_data)} bars for training set.")

        features = compute_features(train_data, {})
        logger.info(f"Number of bars after feature calculation: {len(features)}")

        if len(features) < self.model.sequence_length:
            logger.error(f"Not enough data ({len(features)} bars) to create a single training sequence of length {self.model.sequence_length}. Exiting.")
            raise ValueError("Not enough data for training.")

        # Remove non-numeric columns that aren't suitable for LSTM training
        if 'regime' in features.columns:
            features = features.drop('regime', axis=1)
        if 'prediction' in features.columns:
            features = features.drop('prediction', axis=1)

        # Ensure all features are numeric to prevent scaler errors
        features = features.apply(pd.to_numeric, errors='coerce')
        features = features.dropna()
        logger.info(f"Features after numeric conversion and dropna: {len(features)} bars")

        logger.info(f"Training LSTM model on {len(features)} bars of recent data...")

        # Check if we can load existing model for fine-tuning
        model_exists = os.path.exists(self.cfg["paths"]["model_path"])
        if model_exists:
            try:
                self.model.load(self.cfg["paths"]["model_path"])
                logger.info("Loaded existing model for fine-tuning")
                # Train with fine-tuning enabled
                self.model.train(features, fine_tune=True)
            except Exception as e:
                logger.warning(f"Could not load existing model for fine-tuning: {e}")
                logger.info("Training new model from scratch...")
                self.model.train(features, fine_tune=False)
        else:
            logger.info("No existing model found, training from scratch...")
            self.model.train(features, fine_tune=False)

        self.model.save(self.cfg["paths"]["model_path"])
        logger.info(f"LSTM model training complete. Saved to {self.cfg['paths']['model_path']}")

        # Log model info after training
        model_info = self.model.get_model_info()
        logger.info(f"Model info: {model_info}")

        logger.info("Starting evolutionary search for strategy parameters...")
        evo = EvoSearch(self.cfg)
        best_params = evo.search(features, self.cfg, self.model)
        safe_write_json(self.cfg["paths"]["best_params"], asdict(best_params))
        logger.info(f"Saved best strategy params: {best_params}")

        # Update active_config.json with optimized parameters for live trading
        active_config_path = Path(self.cfg["paths"]["state_dir"]) / "active_config.json"
        active_config = safe_read_json(active_config_path, {})
        if "lstm_strategy_params" not in active_config:
            active_config["lstm_strategy_params"] = {}
        active_config["lstm_strategy_params"].update(asdict(best_params))
        safe_write_json(active_config_path, active_config)
        logger.info(f"Updated active_config.json with optimized parameters")

        return best_params

    def backtest(self, hist: pd.DataFrame, params: LSTMStratParams) -> Dict[str, float]:
        logger.info("Running full backtest with optimized model and parameters...")
        features = compute_features(hist, {})

        try:
            self.model.load(self.cfg["paths"]["model_path"])
            logger.info(f"Loaded model for backtest from {self.cfg['paths']['model_path']}")
        except IOError:
            logger.error("Could not load the model for backtesting. Please run a training cycle first.")
            return {}

        # Set lot_size to 0 to prevent rounding and minimum enforcement in backtest
        lot_size = 0.0

        risk = RiskManager(RiskConfig.from_cfg(self.cfg["risk"]), lot_size=lot_size, fees=self.cfg["fees"])
        strategy = LSTMStrategy(params, risk, self.model, fees=self.cfg["fees"])
        bt = Backtester(self.cfg, strategy)

        metrics = bt.run(features, save_curve=True)
        logger.info(f"Backtest metrics: {metrics}")
        return metrics

    def run_once(self) -> Dict[str, float]:
        hist = self.data.fetch_history(self.cfg["history_bars"])
        logger.info(f"Fetched {len(hist)} total historical bars.")

        now = datetime.now(timezone.utc)
        if not self._last_reopt or (now - self._last_reopt).total_seconds() >= self.cfg["runtime"]["reoptimize_hours"] * 3600:
            logger.info("Re-optimization period reached. Starting new walk-forward cycle.")
            self.params = self.walk_forward_optimize(hist)
            self._last_reopt = now
        else:
            logger.info("Skipping re-optimization, using existing parameters.")
            loaded = safe_read_json(self.cfg["paths"]["best_params"], None)
            if loaded:
                try:
                    self.params = LSTMStratParams(**loaded)
                except TypeError:
                    logger.warning("Saved parameters incompatible, running optimization...")
                    self.params = self.walk_forward_optimize(hist)
                    self._last_reopt = now
            else:
                logger.warning("No existing parameters found. Running optimization anyway.")
                self.params = self.walk_forward_optimize(hist)
                self._last_reopt = now

        metrics = self.backtest(hist, self.params)
        return metrics

    def live_loop(self):
        is_demo = self.cfg.get("runtime", {}).get("demo", True)
        if is_demo and (not API_CLIENT_AVAILABLE or not self.trader.trade):
            logger.error("API client not available or not initialized. Demo trading disabled.")
            return
        elif not is_demo and not CCXT_AVAILABLE:
            logger.error("CCXT library not available. Live trading disabled.")
            return

        logger.info("Starting live trading loop...")
        try:
            self.model.load(self.cfg["paths"]["model_path"])
            # Load parameters from active_config.json as source of truth for live trading
            active_config_path = Path(self.cfg["paths"]["state_dir"]) / "active_config.json"
            active_config = safe_read_json(active_config_path, {})
            loaded_params = active_config.get("lstm_strategy_params", {})
            try:
                self.params = LSTMStratParams(**loaded_params)
                # Force mechanical exit mode for proper trailing stop logic
                self.params.exit_mode = 'mechanical'
            except TypeError:
                converted_params = self._convert_old_params(loaded_params)
                self.params = LSTMStratParams(**converted_params)
                self.params.exit_mode = 'mechanical'
            self.trader.params = self.params
            logger.info("Successfully loaded model and parameters for live trading.")
        except Exception as e:
            logger.error(f"Could not load model/params for live trading. Run training first. Error: {e}")
            return

        creds_present = all([
            bool(os.getenv("YOUR_API_KEY")),
            bool(os.getenv("YOUR_SECRET_KEY")),
            bool(os.getenv("YOUR_PASSPHRASE")),
        ])
        if not creds_present:
            logger.error("API credentials missing. Live trading requires valid keys.")
            return

        logger.info("Fetching initial USDT balance from OKX...")
        balance_response = self.trader.get_balance(ccy='USDT')
        if not balance_response or balance_response.get("code") not in ("0", 0):
            logger.error(f"Balance request failed: {balance_response}")
            return
        if not balance_response.get('data'):
            logger.error("Empty balance response data from OKX.")
            return
        try:
            live_equity = float(balance_response['data'][0]['details'][0]['availBal'])
        except Exception as e:
            logger.error(f"Could not parse live equity from balance response: {e}")
            return
        logger.info(f"Using live available USDT balance for equity: {live_equity:.2f}")

        # Close any existing positions
        logger.info("Checking for existing positions to close...")
        position_response = self.trader.get_position()
        if position_response and position_response.get('code') == '0' and position_response.get('data'):
            for pos in position_response['data']:
                if pos.get('instId') == self.cfg["symbol"] and float(pos.get('pos', 0)) != 0:
                    pos_side = pos.get('posSide', 'net')
                    if pos_side == 'long':
                        self.trader.force_close_position('long')
                    elif pos_side == 'short':
                        self.trader.force_close_position('short')
                    else:
                        # For net mode, close the position
                        self.trader.force_close_position('net')
                    logger.info(f"Closed existing position: {pos}")
        else:
            logger.info("No existing positions found.")

        # Check for positions in other instruments
        logger.info("Checking for positions in other instruments...")
        all_positions_response = self.trader.get_position(instId=None)
        if all_positions_response and all_positions_response.get('code') == '0' and all_positions_response.get('data'):
            other_positions = [pos for pos in all_positions_response['data'] if pos.get('instId') != self.cfg["symbol"] and float(pos.get('pos', 0)) != 0]
            if other_positions:
                for pos in other_positions:
                    logger.info(f"Found position in other instrument: {pos}")
            else:
                logger.info("No positions in other instruments.")
        else:
            logger.info("No positions in other instruments.")

        # Cancel any pending orders
        logger.info("Checking for pending orders to cancel...")
        cancel_result = self.trader.cancel_all_orders(instId=None)
        if cancel_result:
            logger.info(f"Cancelled {cancel_result.get('cancelled', 0)} pending orders")
        else:
            logger.info("No pending orders to cancel or failed to check")

        # Position mode and leverage are already set in trader initialization
        # Only log the current settings
        logger.info("Position mode and leverage already configured in trader initialization")

        if not self.api_client:
            logger.error("API client not initialized. Aborting live mode.")
            return

        instrument_details = self.api_client.market.get_instrument_details(self.cfg["symbol"])
        if not instrument_details or not instrument_details.get("data"):
            logger.error("Failed to fetch instrument details. Aborting live mode.")
            return

        inst_data = instrument_details["data"][0]
        ct_val = float(inst_data.get("ctVal", 1))
        lot_size = float(inst_data.get("lotSz", 1))
        logger.info(f"Instrument details: ctVal={ct_val}, lotSz={lot_size}")

        # Pass lot_size=0 to RiskManager to prevent it from rounding the base currency amount.
        # Rounding should happen on the number of contracts.
        risk = RiskManager(RiskConfig.from_cfg(self.cfg["risk"]),
                           lot_size=0,
                           fees=self.cfg["fees"],
                           live_equity=live_equity)

        sym = self.cfg["symbol"]
        is_demo = self.cfg.get("runtime", {}).get("demo", True)
        if is_demo:
            tf_sub = "candle" + self.cfg["timeframe"].lower()
        else:
            tf_sub = "candle" + self.cfg["timeframe"].lower()
        ws = api_client.OKXWebSocketClient(
            cfg=self.cfg,
            message_queue=queue.Queue(),
            public_subscriptions=[
                {"channel": tf_sub, "instId": sym}
            ],
            private_subscriptions=[
                {"channel": "positions", "instType": "SWAP", "instId": sym},
                {"channel": "account", "ccy": "USDT"}
            ]
        )
        ws.start()
        if not ws.wait_for_ready(timeout=30):
            logger.error("WebSocket connection failed. Aborting live mode.")
            return

        logger.info("Live loop started. Listening for new candles...")
        hist = self.data.fetch_history(self.cfg["history_bars"])
        strategy = LSTMStrategy(self.params, risk, self.model, fees=self.cfg["fees"])
        self.fees = self.cfg["fees"]
        self.entry_px = np.nan
        self.highest_profit = 0.0

        # Initialize trader position tracking
        self.current_positions = []
        self.trader.current_position_side = 0
        self.trader.current_position_size = 0.0
        self.position_levels = 0

        # Re-fetch balance periodically
        balance_update_counter = 0
        BALANCE_UPDATE_INTERVAL = 10  # Update every 10 iterations

        # Online learning - timeframe-based retraining
        def parse_timeframe_to_minutes(timeframe_str):
            """Parse OKX timeframe string to minutes.
            Supports: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h"""
            if timeframe_str.endswith('m'):
                return int(timeframe_str[:-1])
            elif timeframe_str.endswith('h'):
                return int(timeframe_str[:-1]) * 60
            else:
                logger.warning(f"Unknown timeframe format: {timeframe_str}, defaulting to 15m")
                return 15

        # Calculate retraining interval based on configured timeframe
        timeframe_minutes = parse_timeframe_to_minutes(self.cfg["timeframe"])
        RETRAIN_INTERVAL_SECONDS = 100 * timeframe_minutes * 60  # 100 × timeframe duration

        # Initialize retraining tracking
        self.last_retrain_time = getattr(self, 'last_retrain_time', None)
        if self.last_retrain_time is None:
            self.last_retrain_time = datetime.now(timezone.utc)

        # Load or initialize retraining history
        retraining_history_path = Path(self.cfg["paths"]["state_dir"]) / "retraining_history.csv"
        if retraining_history_path.exists():
            retraining_history = pd.read_csv(retraining_history_path).to_dict('records')
        else:
            retraining_history = []

        while True:
            try:
                msg = ws.message_queue.get(timeout=60)
                # Process WebSocket message
                if msg.get('event') == 'subscribe':
                    logger.info("WebSocket subscribed successfully.")
                    continue
                if 'data' in msg and msg['data']:
                    arg = msg.get('arg', {})
                    channel = arg.get('channel')
                    if channel == 'positions':
                        for pos in msg['data']:
                            if pos.get('instId') == sym:
                                pos_side = pos.get('posSide', 'net')
                                prev_positions = self.trader.current_positions.copy()
                                prev_side = self.trader.current_position_side
                                prev_size = self.trader.current_position_size

                                # Update current_positions list
                                side_int = 1 if pos_side == 'long' else -1 if pos_side == 'short' else 0
                                size_str = pos.get('pos', '0')
                                size_float = float(size_str) if size_str else 0.0
                                avg_px_str = pos.get('avgPx', '0') if 'avgPx' in pos else '0'
                                entry_price = float(avg_px_str) if avg_px_str else 0.0
                                level = next((i for i, p in enumerate(self.current_positions) if abs(p['entry_price'] - entry_price) < 1e-6 and p['side'] == side_int), len(self.current_positions))
                                if level == len(self.current_positions):
                                    self.current_positions.append({
                                        'side': side_int,
                                        'size': size_float,
                                        'entry_price': entry_price,
                                        'level': level,
                                        'entry_time': datetime.utcnow()  # Approximate
                                    })
                                else:
                                    self.current_positions[level]['size'] = size_float

                                # Update backward compatible fields
                                total_size = sum(p['size'] for p in self.current_positions)
                                if self.current_positions:
                                    avg_side = max(set(p['side'] for p in self.current_positions), key=lambda s: sum(p['size'] for p in self.current_positions if p['side'] == s))
                                else:
                                    avg_side = 0
                                self.trader.current_position_side = avg_side
                                self.trader.current_position_size = total_size
                                self.position_levels = len(self.current_positions)

                                # Log significant changes
                                if prev_side != self.trader.current_position_side or abs(prev_size - self.trader.current_position_size) > 0.01 or prev_positions != self.trader.current_positions:
                                    logger.info(f"Position updated from WS: side {prev_side}->{self.trader.current_position_side}, size {prev_size:.2f}->{self.trader.current_position_size:.2f}, positions count {len(self.trader.current_positions)}")

                    elif channel == 'account':
                        for acc in msg['data']:
                            if acc.get('ccy') == 'USDT':
                                avail_bal_str = acc.get('availBal', '0')
                                avail_bal = float(avail_bal_str) if avail_bal_str else 0.0
                                logger.info(f"Updated USDT balance from WS: {avail_bal}")
                    elif channel == tf_sub:
                        candle_data = msg['data'][0]
                        # Parse candle data (OKX sends as array: [ts, o, h, l, c, vol])
                        if isinstance(candle_data, list) and len(candle_data) >= 6:
                            ts = int(candle_data[0])
                            o = float(candle_data[1])
                            h = float(candle_data[2])
                            l = float(candle_data[3])
                            c = float(candle_data[4])
                            vol = float(candle_data[5])
                        else:
                            logger.warning(f"Unexpected candle data format: {candle_data}")
                            continue
                        # Create new row
                        new_row = pd.DataFrame({
                            'timestamp': [pd.to_datetime(ts, unit='ms', utc=True)],
                            'open': [o],
                            'high': [h],
                            'low': [l],
                            'close': [c],
                            'volume': [vol]
                        })
                        # Append to historical data
                        hist = pd.concat([hist, new_row], ignore_index=True)
                        # Keep only recent bars
                        hist = hist.tail(self.cfg["history_bars"])
                        # Compute features
                        features = compute_features(hist, {})

                        # Time-based retraining instead of candle counting
                        time_since_retrain = (datetime.now(timezone.utc) - self.last_retrain_time).total_seconds()
                        if time_since_retrain >= RETRAIN_INTERVAL_SECONDS:
                            logger.info(f"Retraining interval reached: {time_since_retrain/3600:.1f} hours (target: {RETRAIN_INTERVAL_SECONDS/3600:.1f} hours)")
                            if len(features) >= self.model.sequence_length:
                                logger.info("Retraining model with live data from correct timeframe...")

                                # Record retraining event
                                retrain_number = len(retraining_history) + 1
                                retraining_history.append({
                                    'timestamp': datetime.now(timezone.utc).isoformat(),
                                    'retrain_number': retrain_number,
                                    'equity_before': live_equity,
                                    'timeframe': self.cfg["timeframe"],
                                    'interval_hours': RETRAIN_INTERVAL_SECONDS / 3600
                                })

                                # Prepare features for training
                                train_features = features.copy()
                                if 'regime' in train_features.columns:
                                    train_features = train_features.drop('regime', axis=1)
                                if 'prediction' in train_features.columns:
                                    train_features = train_features.drop('prediction', axis=1)
                                train_features = train_features.apply(pd.to_numeric, errors='coerce').dropna()
                                if len(train_features) >= self.model.sequence_length:
                                    # Check if we have an existing model to fine-tune
                                    model_exists = os.path.exists(self.cfg["paths"]["model_path"])
                                    if model_exists:
                                        try:
                                            self.model.load(self.cfg["paths"]["model_path"])
                                            logger.info("Loaded existing model for live retraining")
                                            self.model.train(train_features, fine_tune=True)
                                        except Exception as e:
                                            logger.warning(f"Could not load existing model for live retraining: {e}")
                                            self.model.train(train_features, fine_tune=False)
                                    else:
                                        logger.info("No existing model found for live retraining, training from scratch...")
                                        self.model.train(train_features, fine_tune=False)

                                    self.model.save(self.cfg["paths"]["model_path"])

                                    # Save retraining history
                                    pd.DataFrame(retraining_history).to_csv(retraining_history_path, index=False)

                                    # Update plot with retraining data
                                    try:
                                        from scripts.plot_backtest import plot_equity_diagnostics
                                        plot_path = Path(self.cfg["paths"]["state_dir"]) / "backtest_plot.png"
                                        plot_equity_diagnostics(pd.Series([live_equity], index=[pd.Timestamp.now()]), plot_path, None)
                                        logger.info(f"Updated live performance plot with retraining #{retrain_number}")
                                    except Exception as e:
                                        logger.warning(f"Could not update plot: {e}")

                                    logger.info(f"Model retrained and saved (#{retrain_number}). Using {len(train_features)} bars from {self.cfg['timeframe']} timeframe.")
                            self.last_retrain_time = datetime.now(timezone.utc)

                        if len(features) >= self.model.sequence_length:
                                # Manual signal check
                                last_features = features.iloc[-1]
                                prediction = self.model.predict(last_features.to_frame().T)
                                composite = strategy._indicator_signals(last_features.to_frame().T).iloc[-1]
                                adjusted_threshold = self.params.prediction_threshold + 2 * self.fees
                                close_price = last_features['close']

                                # Optional: Skip entry if volatility is too low (to reduce frequent trades)
                                atr_vol = last_features['atr'] / close_price
                                if atr_vol < 0.001:
                                    logger.debug("Low volatility detected (ATR % < 0.1%). Skipping entry check.")
                                    continue

                                if self.trader.current_position_side == 0:
                                    # Check entry
                                    if prediction > adjusted_threshold and composite > 0:
                                        signal = 1
                                    elif prediction < -adjusted_threshold and composite < 0:
                                        signal = -1
                                    else:
                                        signal = 0

                                    if signal != 0:
                                        size_base = risk.get_position_size(features, signal, close_price, position_level=0, current_positions=self.trader.current_positions)

                                        if size_base <= 0:
                                            logger.warning("Calculated position size is zero or negative. Skipping order placement.")
                                            continue

                                        # Convert base currency to contracts/lots
                                        if ct_val > 0:
                                            sz_contracts = size_base / ct_val
                                        else:
                                            logger.error(f"Contract value (ctVal) is zero or invalid: {ct_val}. Cannot calculate order size.")
                                            continue

                                        # Round down to a valid multiple of the instrument's lot size
                                        if lot_size > 0:
                                            sz = math.floor(sz_contracts / lot_size) * lot_size
                                            # Final rounding to handle potential float inaccuracies
                                            sz = round(sz, 8)
                                        else:
                                            sz = sz_contracts

                                        # Ensure we are ordering at least the minimum lot size
                                        if sz < lot_size:
                                            logger.warning(f"Calculated contract size {sz} is less than minimum lot size {lot_size}. Skipping order.")
                                            # Instead of skipping, try to place minimum lot size order if allowed
                                            if lot_size > 0:
                                                sz = lot_size
                                            else:
                                                continue

                                        side = 'buy' if signal == 1 else 'sell'
                                        response = self.trader.place_market(side, str(sz))
                                        if response and response.get('code') == '0':
                                            logger.info(f"Placed {side.upper()} market order for {sz} contracts successfully.")
                                            # Optimistically update state to prevent re-entry before WS update
                                            self.trader.current_position_side = signal
                                            self.trader.current_position_size = size_base
                                            self.trader.current_positions = [{'side': signal, 'size': size_base, 'entry_price': close_price, 'level': 0}]
                                            self.entry_px = close_price
                                            self.highest_profit = 0.0
                                        else:
                                            logger.error(f"{side.upper()} entry failed: {response.get('msg') if response else 'No response'}. Skipping.")

                                else:
                                    # Check exit
                                    logger.debug("Entering exit check branch.")
                                    entry_px = self.trader.current_positions[0]['entry_price'] if self.trader.current_positions else self.entry_px
                                    pnl = (close_price - entry_px) / entry_px if self.trader.current_position_side == 1 else (entry_px - close_price) / entry_px
                                    self.highest_profit = max(self.highest_profit, pnl)

                                    prediction_val = self.model.predict(last_features.to_frame().T)

                                    exit_now = strategy._handle_exit_signals(
                                        last_features,
                                        prediction_val,
                                        close_price,
                                        self.trader.current_position_side,
                                        entry_px,
                                        self.highest_profit,
                                        live_equity
                                    )

                                    if exit_now:
                                        if self.trader.current_position_side == 1:
                                            self.trader.force_close_position('long')
                                        elif self.trader.current_position_side == -1:
                                            self.trader.force_close_position('short')
                                        self.trader.current_position_side = 0
                                        self.trader.current_position_size = 0.0
                                        self.trader.current_positions = []
                                        self.entry_px = np.nan
                                        self.highest_profit = 0.0

            except queue.Empty:
                # Update balance periodically
                balance_update_counter += 1
                if balance_update_counter >= BALANCE_UPDATE_INTERVAL:
                    logger.info("Re-fetching USDT balance...")
                    balance_response = self.trader.get_balance(ccy='USDT')
                    if balance_response and balance_response.get("code") in ("0", 0) and balance_response.get('data'):
                        new_equity = float(balance_response['data'][0]['details'][0]['availBal'])
                        if abs(new_equity - live_equity) > 0.01:
                            logger.info(f"Balance updated: {live_equity:.2f} -> {new_equity:.2f}")
                            live_equity = new_equity
                            risk = RiskManager(RiskConfig.from_cfg(self.cfg["risk"]),
                                               lot_size=lot_size,
                                               fees=self.cfg["fees"],
                                               live_equity=live_equity)
                            strategy = LSTMStrategy(self.params, risk, self.model, fees=self.cfg["fees"])
                        else:
                            logger.debug("Balance unchanged.")
                    balance_update_counter = 0

                # Check for re-optimization
                if (datetime.now(timezone.utc) - (self._last_reopt or datetime.min.replace(tzinfo=timezone.utc))).total_seconds() >= self.cfg["runtime"]["reoptimize_hours"] * 3600:
                    logger.info("Live re-optimization triggered.")
                    try:
                        self.params = self.walk_forward_optimize(hist)
                        self._last_reopt = datetime.now(timezone.utc)
                        self.trader.params = self.params
                        # Update strategy with new params
                        strategy = LSTMStrategy(self.params, risk, self.model, fees=self.cfg["fees"])
                        logger.info("Re-optimization complete.")
                    except Exception as e:
                        logger.error(f"Re-optimization failed: {e}")
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Shutting down live loop.")
                break
            except Exception as e:
                logger.error(f"Error in live loop: {e}")
                time.sleep(5)
