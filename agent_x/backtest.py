"""Backtest the strategy on historical data."""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from .utils import logger
from .strategy import LSTMStrategy

def compute_metrics(equity: pd.Series) -> dict:
    metrics = {}
    if equity.empty:
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDD": 0.0, "Final": 0.0}

    returns = equity.pct_change().dropna()
    if len(returns) == 0:
        metrics.update({"CAGR": 0.0, "Sharpe": 0.0, "MaxDD": 0.0, "Final": float(equity.iloc[-1])})
        return metrics

    days = (equity.index[-1] - equity.index[0]).days
    years = max(days / 365, 1/365)
    base = max(0, equity.iloc[-1]) / equity.iloc[0]
    cagr = base ** (1 / years) - 1

    sharpe = np.sqrt(252) * (returns.mean() / (returns.std() + 1e-12))

    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd = drawdown.min()

    # Underwater metrics
    underwater_mask = equity < roll_max
    total_underwater_loss = ((roll_max - equity) * underwater_mask).sum()
    underwater_bars = underwater_mask.sum()

    metrics.update({
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "MaxDD": float(abs(max_dd)),
        "Final": float(equity.iloc[-1]),
        "Start": float(equity.iloc[0]),
        "TotalUnderwaterLoss": float(total_underwater_loss),
        "UnderwaterBars": int(underwater_bars),
        "TotalBars": len(equity)
    })
    return metrics

class Backtester:
    def __init__(self, cfg, strategy: LSTMStrategy):
        self.cfg = cfg
        self.strategy = strategy
        self.fees = cfg["fees"]
        self.slippage = cfg["slippage"]

    def run(self, df: pd.DataFrame, save_curve=False) -> dict:
        logger.info(f"Starting backtest on {len(df)} bars")

        # Generate all signals first
        main_signals, scaling_signals, exit_signals = self.strategy.generate_signals(df)
        logger.info(f"Generated {len(main_signals)} main signals, {len(scaling_signals)} scaling signals, and {len(exit_signals)} exit signals")

        # Count signal changes
        signal_changes = (main_signals != main_signals.shift(1)).sum()
        logger.info(f"Main signal changes detected: {signal_changes}")

        balance = self.cfg["risk"]["backtest_equity"]
        current_positions = []  # List of dicts: {'side': int, 'size': float, 'entry_price': float, 'level': int}
        equity_curve = []
        trade_history = []

        logger.info(f"Starting backtest with balance: {balance}")

        for i in range(len(df)):
            main_signal = main_signals.iloc[i]
            close_price = df["close"].iloc[i]
            current_ts = df.index[i]

            # Calculate unrealized PnL for all positions
            unrealized_pnl = 0.0
            if current_positions:
                for pos in current_positions:
                    price_diff = close_price - pos['entry_price']
                    unrealized_pnl += price_diff * pos['size'] * pos['side']

            current_equity = balance + unrealized_pnl
            equity_curve.append(current_equity)

            # Check for main signal changes
            if main_signal != (current_positions[0]['side'] if current_positions else 0):
                logger.info(f"Bar {i} ({current_ts}): Main signal change from {(current_positions[0]['side'] if current_positions else 0)} to {main_signal}")

                # Close all existing positions
                for pos in current_positions:
                    if pos['side'] == 1:
                        exit_price = close_price * (1 - self.slippage)
                    else:
                        exit_price = close_price * (1 + self.slippage)

                    price_diff = exit_price - pos['entry_price']
                    realized_pnl = price_diff * pos['size'] * pos['side']

                    notional_entry = pos['size'] * pos['entry_price']
                    notional_exit = pos['size'] * exit_price
                    fees_incurred = self.fees * (notional_entry + notional_exit)

                    net_pnl = realized_pnl - fees_incurred
                    balance += net_pnl

                    trade_history.append({
                        "entry_time": pos.get('entry_time', current_ts),
                        "exit_time": current_ts,
                        "side": "long" if pos['side'] == 1 else "short",
                        "entry_price": pos['entry_price'],
                        "exit_price": exit_price,
                        "size": pos['size'],
                        "level": pos['level'],
                        "gross_pnl": realized_pnl,
                        "fees": fees_incurred,
                        "net_pnl": net_pnl
                    })

                    logger.info(f"CLOSED {('LONG' if pos['side'] == 1 else 'SHORT')} level {pos['level']}: "
                              f"Entry={pos['entry_price']:.2f}, Exit={exit_price:.2f}, "
                              f"Size={pos['size']:.6f}, PnL={net_pnl:.2f}")

                current_positions = []

                # Open new main position
                if main_signal != 0:
                    position_size = self.strategy.risk_manager.get_position_size(df.iloc[:i+1], main_signal, close_price, position_level=0)

                    if position_size > 0:
                        if main_signal == 1:
                            entry_price = close_price * (1 + self.slippage)
                        else:
                            entry_price = close_price * (1 - self.slippage)

                        current_positions.append({
                            'side': main_signal,
                            'size': position_size,
                            'entry_price': entry_price,
                            'level': 0,
                            'entry_time': current_ts
                        })

                        notional_value = entry_price * position_size
                        logger.info(f"OPENED {('LONG' if main_signal == 1 else 'SHORT')} level 0: "
                                  f"Entry={entry_price:.2f}, Size={position_size:.6f}, "
                                  f"Notional=${notional_value:.2f}")

            # Check for exit signals (partial closes)
            if current_positions:
                positions_to_close = []
                for level, exit_signal in enumerate(exit_signals):
                    if exit_signal.iloc[i] == -1:
                        # Find position with this level
                        for pos_idx, pos in enumerate(current_positions):
                            if pos['level'] == level:
                                positions_to_close.append(pos_idx)
                                break

                # Close positions in reverse order (highest level first)
                for pos_idx in sorted(positions_to_close, reverse=True):
                    pos = current_positions[pos_idx]
                    if pos['side'] == 1:
                        exit_price = close_price * (1 - self.slippage)
                    else:
                        exit_price = close_price * (1 + self.slippage)

                    price_diff = exit_price - pos['entry_price']
                    realized_pnl = price_diff * pos['size'] * pos['side']

                    notional_entry = pos['size'] * pos['entry_price']
                    notional_exit = pos['size'] * exit_price
                    fees_incurred = self.fees * (notional_entry + notional_exit)

                    net_pnl = realized_pnl - fees_incurred
                    balance += net_pnl

                    trade_history.append({
                        "entry_time": pos.get('entry_time', current_ts),
                        "exit_time": current_ts,
                        "side": "long" if pos['side'] == 1 else "short",
                        "entry_price": pos['entry_price'],
                        "exit_price": exit_price,
                        "size": pos['size'],
                        "level": pos['level'],
                        "gross_pnl": realized_pnl,
                        "fees": fees_incurred,
                        "net_pnl": net_pnl
                    })

                    logger.info(f"CLOSED PARTIAL {('LONG' if pos['side'] == 1 else 'SHORT')} level {pos['level']}: "
                              f"Entry={pos['entry_price']:.2f}, Exit={exit_price:.2f}, "
                              f"Size={pos['size']:.6f}, PnL={net_pnl:.2f}")

                    # Remove position from list
                    current_positions.pop(pos_idx)

            # Check for scaling signals
            if current_positions:
                main_side = current_positions[0]['side']
                for level, scaling_signal in enumerate(scaling_signals):
                    if scaling_signal.iloc[i] == main_side:
                        # Check if this level already exists
                        existing_levels = [p['level'] for p in current_positions]
                        if (level + 1) not in existing_levels:
                            position_size = self.strategy.risk_manager.get_position_size(df.iloc[:i+1], main_side, close_price, position_level=level+1, current_positions=current_positions)

                            if position_size > 0:
                                if main_side == 1:
                                    entry_price = close_price * (1 + self.slippage)
                                else:
                                    entry_price = close_price * (1 - self.slippage)

                                current_positions.append({
                                    'side': main_side,
                                    'size': position_size,
                                    'entry_price': entry_price,
                                    'level': level + 1,
                                    'entry_time': current_ts
                                })

                                notional_value = entry_price * position_size
                                logger.info(f"OPENED SCALING {('LONG' if main_side == 1 else 'SHORT')} level {level+1}: "
                                          f"Entry={entry_price:.2f}, Size={position_size:.6f}, "
                                          f"Notional=${notional_value:.2f}")

        # Handle final positions
        for pos in current_positions:
            final_price = df["close"].iloc[-1]
            if pos['side'] == 1:
                exit_price = final_price * (1 - self.slippage)
            else:
                exit_price = final_price * (1 + self.slippage)

            price_diff = exit_price - pos['entry_price']
            final_pnl = price_diff * pos['size'] * pos['side']

            notional_entry = pos['size'] * pos['entry_price']
            notional_exit = pos['size'] * exit_price
            fees_incurred = self.fees * (notional_entry + notional_exit)

            net_pnl = final_pnl - fees_incurred
            balance += net_pnl

            logger.info(f"CLOSED FINAL POSITION level {pos['level']}: PnL={net_pnl:.2f}")

        equity_series = pd.Series(equity_curve, index=df.index)
        metrics = compute_metrics(equity_series)

        logger.info(f"Backtest completed:")
        logger.info(f"  Total trades: {len(trade_history)}")
        logger.info(f"  Final balance: {balance:.2f} (started: {self.cfg['risk']['backtest_equity']:.2f})")
        logger.info(f"  Total return: {((balance / self.cfg['risk']['backtest_equity']) - 1) * 100:.2f}%")

        if save_curve:
            equity_df = pd.DataFrame({"ts": equity_series.index, "equity": equity_series.values})
            equity_path = Path(self.cfg["paths"]["equity_curve"])
            equity_df.to_csv(equity_path, index=False)
            logger.info(f"Saved equity curve to {equity_path}")

            if trade_history:
                trade_path = Path(self.cfg["paths"].get("trade_history", "state/trade_history.csv"))
                pd.DataFrame(trade_history).to_csv(trade_path, index=False)
                logger.info(f"Saved {len(trade_history)} trades to {trade_path}")

            try:
                from scripts.plot_backtest import plot_equity_diagnostics
                plot_path = equity_path.with_name("backtest_plot.png")
                plot_equity_diagnostics(equity_series, plot_path, trade_path)
            except ImportError:
                logger.warning("Could not import plot_equity_diagnostics")

        return metrics
