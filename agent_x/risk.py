from dataclasses import dataclass
import math
import pandas as pd
from .utils import logger

@dataclass
class RiskConfig:
    backtest_equity: float
    vol_target: float
    max_leverage: int
    max_daily_loss: float
    max_exposure: float
    risk_per_trade: float
    max_open_orders: int = 1
 
    @classmethod
    def from_cfg(cls, cfg: dict):
        valid_keys = {
            'backtest_equity', 'vol_target', 'max_leverage',
            'max_daily_loss', 'max_exposure', 'risk_per_trade', 'max_open_orders'
        }
        filtered = {k: v for k, v in cfg.items() if k in valid_keys}
        return cls(**filtered)

class RiskManager:
    def __init__(self, config: RiskConfig, lot_size: float = 0.001, allow_min_fill: bool = True, fees: float = 0.0005, live_equity: float = None):
        self.config = config
        self.lot_size = float(lot_size) if lot_size else 0.0
        self.allow_min_fill = allow_min_fill
        self.fees = fees
        self.equity = live_equity if live_equity is not None else self.config.backtest_equity

    def _cap_by_leverage_units(self, close: float) -> float:
        if close is None or close <= 0:
            return 0.0
        notional_cap = self.equity * self.config.max_leverage
        return max(0.0, notional_cap / close)

    def _cap_by_exposure_units(self, close: float) -> float:
        if close is None or close <= 0:
            return 0.0
        notional_cap = self.equity * self.config.max_exposure
        return max(0.0, notional_cap / close)

    def _risk_based_units(self, atr: float, close: float) -> float:
        if close is None or close <= 0:
            return 0.0
        if atr is None or atr <= 0:
            return 0.0  # Changed from float('inf') to 0.0 to avoid ignoring risk_per_trade and defaulting to min lot size
        risk_amount = self.equity * self.config.risk_per_trade
        units = risk_amount / atr
        return max(0.0, units)

    def get_position_size(self, df: pd.DataFrame, signal: int, close: float, position_level: int = 0, current_positions: list = None) -> float:
        if signal == 0:
            return 0.0

        # Calculate current total exposure
        current_exposure = 0.0
        if current_positions:
            for pos in current_positions:
                current_exposure += pos['size'] * pos['entry_price']

        atr = None
        if isinstance(df, pd.DataFrame) and len(df) > 0 and 'atr' in df.columns:
            try:
                atr = float(df.iloc[-1]['atr'])
            except Exception:
                atr = None

        # Early return if no valid ATR to ensure risk_per_trade is respected
        if atr is None or atr <= 0:
            logger.warning(f"RiskManager: No valid ATR available (atr={atr}), skipping position to respect risk_per_trade.")
            return 0.0

        units_cap_lev = self._cap_by_leverage_units(close)
        units_cap_exp = self._cap_by_exposure_units(close)
        units_cap_risk = self._risk_based_units(atr, close)

        # Adjust exposure cap for remaining capacity
        remaining_exposure = self.equity * self.config.max_exposure - current_exposure
        if remaining_exposure > 0:
            units_cap_exp_remaining = max(0.0, remaining_exposure / close)
            units_cap_exp = min(units_cap_exp, units_cap_exp_remaining)
        else:
            # No remaining exposure capacity
            return 0.0

        raw_units = min(units_cap_lev, units_cap_exp, units_cap_risk)

        # Apply scaling ratios for multiple positions
        scaling_ratios = [1.0, 0.5, 0.25]  # Initial, second, third position ratios
        if position_level < len(scaling_ratios):
            raw_units *= scaling_ratios[position_level]
        else:
            raw_units *= 0.25  # Default for additional levels

        # Dynamic risk adjustment for small capital strategy
        raw_units = self._apply_dynamic_risk_adjustment(df, raw_units, close)

        # Adjust for estimated round-trip fees (entry + exit)
        fee_adjustment_factor = 1 - 2 * self.fees
        raw_units *= fee_adjustment_factor

        # Micro-position scaling for small capital
        raw_units = self._scale_position_for_small_cap(raw_units, close)

        # Enforce a hard minimum position size of 1 lot size only if raw_units > 0
        if raw_units > 0 and raw_units < self.lot_size:
            if self.allow_min_fill:
                logger.debug(f"RiskManager: raw_units {raw_units:.6f} below lot_size {self.lot_size}, enforcing minimum lot size")
                raw_units = self.lot_size
            else:
                logger.debug(f"RiskManager: raw_units {raw_units:.6f} below lot_size {self.lot_size}, skipping (allow_min_fill=False)")
                return 0.0

        if self.lot_size > 0:
            rounded_units = math.floor(raw_units / self.lot_size) * self.lot_size
            rounded_units = round(rounded_units, 8)  # Round to 8 decimal places to avoid floating point issues
        else:
            rounded_units = raw_units

        if rounded_units < self.lot_size and self.lot_size > 0:
            logger.debug(f"RiskManager: rounded_units {rounded_units:.6f} below lot_size {self.lot_size}, cannot min-fill")
            return 0.0

        from dataclasses import dataclass
import math
import pandas as pd
from .utils import logger

@dataclass
class RiskConfig:
    backtest_equity: float
    leverage: int
    max_daily_loss: float
    max_exposure: float
    risk_per_trade: float
    max_open_orders: int = 1

    @classmethod
    def from_cfg(cls, cfg: dict):
        valid_keys = {
            'backtest_equity', 'leverage',
            'max_daily_loss', 'max_exposure', 'risk_per_trade', 'max_open_orders'
        }
        filtered = {k: v for k, v in cfg.items() if k in valid_keys}
        return cls(**filtered)

class RiskManager:
    def __init__(self, config: RiskConfig, lot_size: float = 0.001, allow_min_fill: bool = True, fees: float = 0.0005, live_equity: float = None):
        self.config = config
        self.lot_size = float(lot_size) if lot_size else 0.0
        self.allow_min_fill = allow_min_fill
        self.fees = fees
        self.equity = live_equity if live_equity is not None else self.config.backtest_equity

    def _cap_by_leverage_units(self, close: float) -> float:
        if close is None or close <= 0:
            return 0.0
        notional_cap = self.equity * self.config.leverage
        return max(0.0, notional_cap / close)

    def _cap_by_exposure_units(self, close: float) -> float:
        if close is None or close <= 0:
            return 0.0
        notional_cap = self.equity * self.config.max_exposure
        return max(0.0, notional_cap / close)

    def _risk_based_units(self, atr: float, close: float) -> float:
        if close is None or close <= 0:
            return 0.0
        if atr is None or atr <= 0:
            return 0.0  # Changed from float('inf') to 0.0 to avoid ignoring risk_per_trade and defaulting to min lot size
        risk_amount = self.equity * self.config.risk_per_trade
        units = risk_amount / atr
        return max(0.0, units)

    def get_position_size(self, df: pd.DataFrame, signal: int, close: float, position_level: int = 0, current_positions: list = None) -> float:
        if signal == 0:
            return 0.0

        # Calculate current total exposure
        current_exposure = 0.0
        if current_positions:
            for pos in current_positions:
                current_exposure += pos['size'] * pos['entry_price']

        atr = None
        if isinstance(df, pd.DataFrame) and len(df) > 0 and 'atr' in df.columns:
            try:
                atr = float(df.iloc[-1]['atr'])
            except Exception:
                atr = None

        # Early return if no valid ATR to ensure risk_per_trade is respected
        if atr is None or atr <= 0:
            logger.warning(f"RiskManager: No valid ATR available (atr={atr}), skipping position to respect risk_per_trade.")
            return 0.0

        units_cap_lev = self._cap_by_leverage_units(close)
        units_cap_exp = self._cap_by_exposure_units(close)
        units_cap_risk = self._risk_based_units(atr, close)

        # Adjust exposure cap for remaining capacity
        remaining_exposure = self.equity * self.config.max_exposure - current_exposure
        if remaining_exposure > 0:
            units_cap_exp_remaining = max(0.0, remaining_exposure / close)
            units_cap_exp = min(units_cap_exp, units_cap_exp_remaining)
        else:
            # No remaining exposure capacity
            return 0.0

        raw_units = min(units_cap_lev, units_cap_exp, units_cap_risk)

        # Apply scaling ratios for multiple positions
        scaling_ratios = [1.0, 0.5, 0.25]  # Initial, second, third position ratios
        if position_level < len(scaling_ratios):
            raw_units *= scaling_ratios[position_level]
        else:
            raw_units *= 0.25  # Default for additional levels

        # Adjust for estimated round-trip fees (entry + exit)
        fee_adjustment_factor = 1 - 2 * self.fees
        raw_units *= fee_adjustment_factor

        # Enforce a hard minimum position size of 1 lot size only if raw_units > 0
        if raw_units > 0 and raw_units < self.lot_size:
            if self.allow_min_fill:
                logger.debug(f"RiskManager: raw_units {raw_units:.6f} below lot_size {self.lot_size}, enforcing minimum lot size")
                raw_units = self.lot_size
            else:
                logger.debug(f"RiskManager: raw_units {raw_units:.6f} below lot_size {self.lot_size}, skipping (allow_min_fill=False)")
                return 0.0

        if self.lot_size > 0:
            rounded_units = math.floor(raw_units / self.lot_size) * self.lot_size
            rounded_units = round(rounded_units, 8)  # Round to 8 decimal places to avoid floating point issues
        else:
            rounded_units = raw_units

        if rounded_units < self.lot_size and self.lot_size > 0:
            logger.debug(f"RiskManager: rounded_units {rounded_units:.6f} below lot_size {self.lot_size}, cannot min-fill")
            return 0.0

        logger.debug(
            f"RiskManager sizing: close={close:.6f}, atr={atr if atr is not None else float('nan')}, "
            f"units_cap_lev={units_cap_lev:.6f}, units_cap_exp={units_cap_exp:.6f}, "
            f"units_cap_risk={units_cap_risk if units_cap_risk != float('inf') else -1:.6f}, "
            f"raw_units={raw_units:.6f} (pre-fee), final_units={rounded_units:.6f}, "
            f"current_exposure={current_exposure:.2f}, remaining_exposure={remaining_exposure:.2f}"
        )
        return rounded_units

    def _apply_dynamic_risk_adjustment(self, df: pd.DataFrame, base_units: float, close: float) -> float:
        """Apply dynamic risk adjustment based on recent performance and equity growth"""
        # Calculate recent performance for Kelly criterion
        recent_returns = self._calculate_recent_performance(df)
        kelly_fraction = self._kelly_criterion(recent_returns)

        # Scale risk with equity growth (for small capital strategy)
        equity_multiplier = min(self.equity / 25.0, 10.0)  # Cap at 10x starting capital
        adjusted_risk = equity_multiplier  # Removed kelly_fraction to prevent over-conservative sizing; now only uses equity growth multiplier

        return base_units * adjusted_risk

    def _calculate_recent_performance(self, df: pd.DataFrame) -> list:
        """Calculate recent trade performance for Kelly criterion"""
        if len(df) < 50:
            return [0.01]  # Default small positive return

        # Simple approximation: use recent price changes as performance proxy
        recent_prices = df['close'].tail(50)
        returns = recent_prices.pct_change().dropna().tolist()
        return returns[-20:] if len(returns) > 20 else returns

    def _kelly_criterion(self, returns: list) -> float:
        """Calculate Kelly fraction based on recent returns"""
        if not returns:
            return 0.5  # Conservative default

        # Simplified Kelly calculation
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        avg_win = sum(r for r in returns if r > 0) / max(1, sum(1 for r in returns if r > 0))
        avg_loss = abs(sum(r for r in returns if r < 0) / max(1, sum(1 for r in returns if r < 0)))

        if avg_loss == 0:
            return 0.5

        kelly = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        kelly = max(0.1, min(kelly, 1.0))  # Bound between 0.1 and 1.0

        return kelly

    def _scale_position_for_small_cap(self, base_size: float, close: float) -> float:
        """Scale positions to ensure minimum viable trade sizes for small capital"""
        min_notional = 1.0  # Minimum $1 trade
        max_position_pct = 0.8  # Max 80% of equity per trade

        scaled_size = min(base_size, self.equity * max_position_pct / close)
        min_size = min_notional / close

        return max(scaled_size, min_size)

        return rounded_units

    def _apply_dynamic_risk_adjustment(self, df: pd.DataFrame, base_units: float, close: float) -> float:
        """Apply dynamic risk adjustment based on recent performance and equity growth"""
        # Calculate recent performance for Kelly criterion
        recent_returns = self._calculate_recent_performance(df)
        kelly_fraction = self._kelly_criterion(recent_returns)

        # Scale risk with equity growth (for small capital strategy)
        equity_multiplier = min(self.equity / 25.0, 10.0)  # Cap at 10x starting capital
        adjusted_risk = equity_multiplier  # Removed kelly_fraction to prevent over-conservative sizing; now only uses equity growth multiplier

        return base_units * adjusted_risk

    def _calculate_recent_performance(self, df: pd.DataFrame) -> list:
        """Calculate recent trade performance for Kelly criterion"""
        if len(df) < 50:
            return [0.01]  # Default small positive return

        # Simple approximation: use recent price changes as performance proxy
        recent_prices = df['close'].tail(50)
        returns = recent_prices.pct_change().dropna().tolist()
        return returns[-20:] if len(returns) > 20 else returns

    def _kelly_criterion(self, returns: list) -> float:
        """Calculate Kelly fraction based on recent returns"""
        if not returns:
            return 0.5  # Conservative default

        # Simplified Kelly calculation
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        avg_win = sum(r for r in returns if r > 0) / max(1, sum(1 for r in returns if r > 0))
        avg_loss = abs(sum(r for r in returns if r < 0) / max(1, sum(1 for r in returns if r < 0)))

        if avg_loss == 0:
            return 0.5

        kelly = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        kelly = max(0.1, min(kelly, 1.0))  # Bound between 0.1 and 1.0

        return kelly

    def _scale_position_for_small_cap(self, base_size: float, close: float) -> float:
        """Scale positions to ensure minimum viable trade sizes for small capital"""
        min_notional = 1.0  # Minimum $1 trade
        max_position_pct = 0.8  # Max 80% of equity per trade

        scaled_size = min(base_size, self.equity * max_position_pct / close)
        min_size = min_notional / close

        return max(scaled_size, min_size)
