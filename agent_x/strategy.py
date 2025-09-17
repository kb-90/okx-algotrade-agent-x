import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from .risk import RiskManager
from .model import LSTMModel
from .utils import logger
import json
from pathlib import Path

@dataclass
class LSTMStratParams:
    prediction_threshold: float = 0.005  # Lowered for more entries
    indicator_weights: dict = field(default_factory=lambda: {'rsi': 0.400008272490776, 'macd': 0.36285517542429074, 'bb': 0.2371365520849334})  # Updated to best_params
    exit_mode: str = 'mechanical'  # 'mechanical' or 'intelligent'
    atr_mult_sl: float = 3.0  # Lowered for tighter stops
    initial_trail_pct: float = 0.05  # Updated to best_params
    profit_trigger_pct: float = 0.3  # Increased for higher profit targets
    tighter_trail_pct: float = 0.01  # Updated to best_params
    lstm_disagreement_pct: float = 0.03  # Updated to best_params
    future_bars: int = 9  # From config, updated to best_params
    vol_target: float = 0.7  # From risk config, updated to best_params
    max_daily_loss: float = 0.15  # From risk config, updated to best_params
    max_exposure: float = 0.7  # From risk config, updated to best_params
    risk_per_trade: float = 0.25  # From risk config, updated to best_params
    scaling_thresholds: list = field(default_factory=lambda: [0.01, 0.02])  # Thresholds for scaling positions (1% and 2%)
    partial_exit_thresholds: list = field(default_factory=lambda: [0.005, 0.01, 0.015])  # Thresholds for partial exits (0.5%, 1%, 1.5%)

    def __post_init__(self):
        # Helper to coerce floats safely
        def _f(x, default):
            try:
                v = float(x)
                return v
            except Exception:
                return default

        # Try to load best_params.json if it exists
        best_params_path = Path("state/best_params.json")
        if best_params_path.exists():
            try:
                with open(best_params_path, 'r') as f:
                    best_params = json.load(f)
                self.prediction_threshold = best_params.get('prediction_threshold', self.prediction_threshold)
                self.indicator_weights = best_params.get('indicator_weights', self.indicator_weights)
                self.exit_mode = best_params.get('exit_mode', self.exit_mode)
                self.atr_mult_sl = best_params.get('atr_mult_sl', self.atr_mult_sl)
                self.initial_trail_pct = best_params.get('initial_trail_pct', self.initial_trail_pct)
                self.profit_trigger_pct = best_params.get('profit_trigger_pct', self.profit_trigger_pct)
                self.tighter_trail_pct = best_params.get('tighter_trail_pct', self.tighter_trail_pct)
                self.lstm_disagreement_pct = best_params.get('lstm_disagreement_pct', self.lstm_disagreement_pct)
                self.future_bars = best_params.get('future_bars', self.future_bars)
                self.vol_target = best_params.get('vol_target', self.vol_target)
                self.max_daily_loss = best_params.get('max_daily_loss', self.max_daily_loss)
                self.max_exposure = best_params.get('max_exposure', self.max_exposure)
                self.risk_per_trade = best_params.get('risk_per_trade', self.risk_per_trade)
                self.scaling_thresholds = best_params.get('scaling_thresholds', self.scaling_thresholds)
                self.partial_exit_thresholds = best_params.get('partial_exit_thresholds', self.partial_exit_thresholds)
            except Exception as e:
                logger.warning(f"Failed to load best_params.json: {e}")

        # Coerce/keep provided values, fill sensible defaults if falsy/invalid
        self.indicator_weights = self.indicator_weights or {'rsi': 0.400008272490776, 'macd': 0.36285517542429074, 'bb': 0.2371365520849334}
        self.prediction_threshold = _f(self.prediction_threshold, 0.002)
        self.exit_mode = self.exit_mode or 'intelligent'
        self.atr_mult_sl = _f(self.atr_mult_sl, 3.0)
        self.initial_trail_pct = _f(self.initial_trail_pct, 0.05)
        self.profit_trigger_pct = _f(self.profit_trigger_pct, 0.3)
        self.tighter_trail_pct = _f(self.tighter_trail_pct, 0.01)
        self.lstm_disagreement_pct = _f(self.lstm_disagreement_pct, 0.015)
        self.future_bars = int(self.future_bars) if self.future_bars else 9
        self.vol_target = _f(self.vol_target, 0.7)
        self.max_daily_loss = _f(self.max_daily_loss, 0.15)
        self.max_exposure = _f(self.max_exposure, 0.7)
        self.risk_per_trade = _f(self.risk_per_trade, 0.25)
        # Note: scaling_thresholds and partial_exit_thresholds are lists, handled separately

class LSTMStrategy:
    def __init__(self, params: LSTMStratParams, risk_manager: RiskManager, model: LSTMModel, fees: float = 0.0005):
        self.params = params
        self.risk_manager = risk_manager
        self.model = model
        self.fees = fees

    def _require_columns(self, df: pd.DataFrame, cols: list[str]) -> None:
        """Check if required columns exist in DataFrame"""
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

    def _indicator_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute composite indicator signals from RSI, MACD, and Bollinger Bands.

        Logic:
        - RSI: Oversold (<30) = bullish signal (1), Overbought (>70) = bearish signal (-1)
        - MACD: Above signal line = bullish (1), Below = bearish (-1)
        - BB: Price below lower band = bullish (1), Above upper band = bearish (-1)

        Returns weighted composite signal between -1 (bearish) and 1 (bullish).
        """
        self._require_columns(df, ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'close'])
        rsi = df['rsi']
        # RSI signals: Oversold indicates potential reversal up, overbought indicates potential reversal down
        rsi_sig = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))

        macd = df['macd']
        macd_sigl = df['macd_signal']
        # MACD signals: Crossover above signal line suggests momentum up, below suggests momentum down
        macd_sig = np.where(macd > macd_sigl, 1, np.where(macd < macd_sigl, -1, 0))

        close = df['close']
        bb_u = df['bb_upper']
        bb_l = df['bb_lower']
        # Bollinger Band signals: Price touching lower band suggests oversold bounce, upper band suggests overbought pullback
        bb_sig = np.where(close < bb_l, 1, np.where(close > bb_u, -1, 0))

        w = self.params.indicator_weights or {'rsi': 0.33, 'macd': 0.33, 'bb': 0.34}
        composite = (rsi_sig * w.get('rsi', 0.33) +
                     macd_sig * w.get('macd', 0.33) +
                     bb_sig * w.get('bb', 0.34))
        return pd.Series(composite, index=df.index)

    def _handle_entry_signals(self, dfc: pd.DataFrame, i: int, px: float) -> int:
        """Determine entry signal at index i."""
        # Volatility check
        if (dfc['atr'].iloc[i] / px) < 0.001:
            return 0

        fee_estimate = 2 * self.fees  # Round-trip fee estimate
        adjusted_threshold = float(self.params.prediction_threshold) + fee_estimate

        pred_rel = dfc['prediction'].astype(float)
        composite = self._indicator_signals(dfc)

        long_entry = (pred_rel > adjusted_threshold) & (composite > 0)
        short_entry = (pred_rel < -adjusted_threshold) & (composite < 0)

        if long_entry.iloc[i]:
            logger.debug(f"Bar {i}: LONG entry signal - Prediction: {pred_rel.iloc[i]:.6f}, Composite: {composite.iloc[i]:.6f}")
            return 1
        elif short_entry.iloc[i]:
            logger.debug(f"Bar {i}: SHORT entry signal - Prediction: {pred_rel.iloc[i]:.6f}, Composite: {composite.iloc[i]:.6f}")
            return -1
        else:
            logger.debug(f"Bar {i}: No entry signal - Prediction: {pred_rel.iloc[i]:.6f}, Composite: {composite.iloc[i]:.6f}")
            return 0

    def _handle_exit_signals(self, dfc: pd.Series, prediction: float, px: float, pos: int, entry_px: float, highest_profit: float, equity: Optional[float] = None) -> bool:
        """
        Determine if an exit signal should be triggered.
        `dfc` is now a Series (a single row of the dataframe).
        `prediction` is the single float prediction for the current step.
        """
        pnl = (px - entry_px) / entry_px if pos == 1 else (entry_px - px) / entry_px

        exit_now = False

        if self.params.exit_mode == 'mechanical':
            if 'atr' in dfc and not np.isnan(dfc['atr']):
                atr = float(dfc['atr'])
                if atr > 0 and entry_px > 0:
                    atr_stop = (self.params.atr_mult_sl * atr) / entry_px
                    if pnl < -abs(atr_stop):
                        logger.debug(f"ATR stop exit - PnL: {pnl:.4f}, ATR stop: {atr_stop:.4f}")
                        exit_now = True

            if pnl < -abs(self.params.initial_trail_pct):
                logger.debug(f"Initial trail stop exit - PnL: {pnl:.4f}, Threshold: {self.params.initial_trail_pct:.4f}")
                exit_now = True

            if highest_profit >= self.params.profit_trigger_pct:
                if pnl < highest_profit - self.params.tighter_trail_pct:
                    logger.debug(f"Tighter trail exit - PnL: {pnl:.4f}, Highest: {highest_profit:.4f}")
                    exit_now = True
            elif highest_profit >= self.params.initial_trail_pct:
                if pnl < highest_profit - self.params.initial_trail_pct:
                    logger.debug(f"Initial trail exit - PnL: {pnl:.4f}, Highest: {highest_profit:.4f}")
                    exit_now = True

        elif self.params.exit_mode == 'intelligent':
            if pnl > highest_profit:
                highest_profit = pnl
            if pnl < highest_profit - self.params.initial_trail_pct:
                return True
            
            disagree = self.params.lstm_disagreement_pct
            prediction_pct = (prediction - px) / px
            if pos == 1 and prediction_pct < -abs(disagree):
                logger.debug(f"LSTM disagreement exit (LONG) - Prediction Pct: {prediction_pct:.6f}, Threshold: {disagree:.6f}")
                exit_now = True
            elif pos == -1 and prediction_pct > abs(disagree):
                logger.debug(f"LSTM disagreement exit (SHORT) - Prediction Pct: {prediction_pct:.6f}, Threshold: {disagree:.6f}")
                exit_now = True

        # Additional exit logic for small capital
        if not exit_now:
            if equity is None:
                try:
                    equity = self.risk_manager.equity
                except AttributeError:
                    equity = 50.0  # Default fallback

            if equity < 50.0:
                if pnl < -0.025: exit_now = True
                if pnl > 0.05: exit_now = True
            else:
                if pnl < -0.05: exit_now = True
                if pnl > 0.075: exit_now = True

        return exit_now

    def generate_signals(self, df: pd.DataFrame, current_positions: list = None, equity: Optional[float] = None) -> tuple:
        """
        Generate signals for main position and scaling positions.
        """
        if df.empty:
            return pd.Series(dtype=int), [], []

        dfc = df.copy()
        self._require_columns(dfc, ['close'])

        if 'prediction' not in dfc.columns or dfc['prediction'].isna().all():
            preds = self.model.predict_sequence(dfc)
            if len(preds) < len(dfc):
                pad = np.full(len(dfc) - len(preds), np.nan)
                preds = np.concatenate((pad, preds))
            dfc['prediction_val'] = pd.Series(preds, index=dfc.index)
            dfc['prediction'] = (dfc['prediction_val'] - dfc['close']) / dfc['close']
        else:
            # Ensure prediction_val exists if prediction does
            if 'prediction_val' not in dfc.columns:
                 dfc['prediction_val'] = (dfc['prediction'] * dfc['close']) + dfc['close']

        main_signals = np.zeros(len(dfc), dtype=int)
        scaling_signals = [np.zeros(len(dfc), dtype=int) for _ in range(self.risk_manager.config.max_open_orders - 1)]
        exit_signals = [np.zeros(len(dfc), dtype=int) for _ in range(self.risk_manager.config.max_open_orders)]

        pos = 0
        entry_px = np.nan
        highest_profit = 0.0
        position_levels = 0

        for i in range(len(dfc)):
            px = float(dfc['close'].iloc[i])
            current_row = dfc.iloc[i]
            prediction_val = current_row['prediction_val']

            if pos == 0:
                main_signals[i] = self._handle_entry_signals(dfc, i, px)
                if main_signals[i] != 0:
                    pos = main_signals[i]
                    entry_px = px
                    highest_profit = 0.0
                    position_levels = 1
            else:
                exit_signal = self._handle_exit_signals(current_row, prediction_val, px, pos, entry_px, highest_profit, equity)
                if exit_signal:
                    main_signals[i] = 0
                    pos = 0
                    entry_px = np.nan
                    highest_profit = 0.0
                    position_levels = 0
                    for level in range(self.risk_manager.config.max_open_orders):
                        exit_signals[level][i] = -1
                else:
                    main_signals[i] = pos
                    if position_levels < self.risk_manager.config.max_open_orders:
                        price_change = (px - entry_px) / entry_px if pos == 1 else (entry_px - px) / entry_px
                        scaling_thresholds = self.params.scaling_thresholds
                        for level in range(1, self.risk_manager.config.max_open_orders):
                            if position_levels == level and price_change >= scaling_thresholds[level - 1]:
                                scaling_signals[level - 1][i] = pos
                                position_levels += 1
                                break

                    if position_levels > 1:
                        current_pnl = (px - entry_px) / entry_px if pos == 1 else (entry_px - px) / entry_px
                        threshold_3 = self.params.partial_exit_thresholds[2] if hasattr(self.params, 'partial_exit_thresholds') and len(self.params.partial_exit_thresholds) > 2 else 0.01
                        if position_levels >= 3 and current_pnl < threshold_3:
                            exit_signals[2][i] = -1
                            position_levels -= 1
                        threshold_2 = self.params.partial_exit_thresholds[1] if hasattr(self.params, 'partial_exit_thresholds') and len(self.params.partial_exit_thresholds) > 1 else 0.005
                        if position_levels >= 2 and current_pnl < threshold_2:
                            exit_signals[1][i] = -1
                            position_levels -= 1

            if pos != 0:
                pnl = (px - entry_px) / entry_px if pos == 1 else (entry_px - px) / entry_px
                if pnl > highest_profit:
                    highest_profit = pnl

        return (pd.Series(main_signals, index=df.index, dtype=int),
                [pd.Series(sig, index=df.index, dtype=int) for sig in scaling_signals],
                [pd.Series(sig, index=df.index, dtype=int) for sig in exit_signals])
