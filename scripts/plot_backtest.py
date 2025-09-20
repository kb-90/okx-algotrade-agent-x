"""Plot backtest equity curve and diagnostics.

Reads: state/equity_curve.csv (columns: ts,equity)
Writes: state/backtest_plot.png

Usage: python scripts/plot_backtest.py [input_csv] [output_png] [trade_history_csv]
"""
from __future__ import annotations
import sys
import traceback
try:
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    print(f"An error occurred during import: {e}")
    traceback.print_exc()
    sys.exit(1)


def load_equity(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'])
        df = df.set_index('ts')
    elif df.shape[1] >= 2:
        df.columns = ['ts', 'equity']
        df['ts'] = pd.to_datetime(df['ts'])
        df = df.set_index('ts')
    else:
        raise ValueError('CSV must contain timestamp and equity columns')
    # ensure timezone-aware handled by pandas
    series = df['equity'].astype(float)
    series.index = pd.to_datetime(series.index)
    return series


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

    metrics.update({
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "MaxDD": float(abs(max_dd)),
        "Final": float(equity.iloc[-1]),
        "Start": float(equity.iloc[0])
    })
    return metrics


def plot_equity_diagnostics(equity: pd.Series, out_path: Path, trade_history_path: Path = None):
    # compute
    returns = equity.pct_change().dropna()
    metrics = compute_metrics(equity)
    rolling = equity.pct_change().rolling(window=6, min_periods=1).mean()  # example rolling

    fig = plt.figure(constrained_layout=True, figsize=(14, 9))
    gs = fig.add_gridspec(3, 4)

    ax_eq = fig.add_subplot(gs[0:2, :])  # big top plot
    ax_dd = fig.add_subplot(gs[2, 0:2])
    ax_hist = fig.add_subplot(gs[2, 2:4])

    # Equity curve
    ax_eq.plot(equity.index.to_numpy(), equity.to_numpy(), label='Equity', color='tab:blue')
    ax_eq.fill_between(equity.index.to_numpy(), equity.to_numpy(), equity.cummax().to_numpy(), where=(equity < equity.cummax()),
                       color='tab:red', alpha=0.08, interpolate=True, label='Underwater')
    ax_eq.set_title('Historical Trades || Equity Curve || Agent-X - by Kevin Bourn')
    ax_eq.set_ylabel('Equity')
    ax_eq.grid(alpha=0.3)

    # Plot entry and exit markers
    if trade_history_path and trade_history_path.exists():
        trade_history = pd.read_csv(trade_history_path)
        if 'entry_time' in trade_history.columns:
            trade_history['entry_time'] = pd.to_datetime(trade_history['entry_time'], errors='coerce')
            trade_history['exit_time'] = pd.to_datetime(trade_history['exit_time'], errors='coerce')

            for _, trade in trade_history.iterrows():
                entry_time = trade['entry_time']
                exit_time = trade['exit_time']
                entry_equity = equity.asof(entry_time)
                exit_equity = equity.asof(exit_time)

                if trade['level'] == 0:
                    if trade['side'] == 'long':
                        ax_eq.plot(entry_time, entry_equity, '^', color='green', markersize=8, label='Long Entry')
                    else:
                        ax_eq.plot(entry_time, entry_equity, 'v', color='red', markersize=8, label='Short Entry')
                else:
                    if trade['side'] == 'long':
                        ax_eq.plot(entry_time, entry_equity, '^', color='yellow', markersize=6, label='Long Scale Entry')
                    else:
                        ax_eq.plot(entry_time, entry_equity, 'v', color='orange', markersize=6, label='Short Scale Entry')
                
                ax_eq.plot(exit_time, exit_equity, 'o', color='blue', markersize=6, label='Exit')

                ax_eq.plot([entry_time, exit_time], [entry_equity, exit_equity], linestyle=':', color='gray', linewidth=1)
        # If not backtest format, skip trade plotting for live mode

    # annotate metrics box
    text = (
        f"Start: {metrics['Start']:.2f}\n"
        f"Final: {metrics['Final']:.2f}\n"
        f"CAGR: {metrics['CAGR']*100:.2f}%\n"
        f"Sharpe: {metrics['Sharpe']:.2f}\n"
        f"MaxDD: {metrics['MaxDD']*100:.2f}%\n"
    )
    ax_eq.text(0.99, 0.02, text, transform=ax_eq.transAxes, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # rolling returns overlay (secondary axis)
    ax_r = ax_eq.twinx()
    ax_r.plot(rolling.index.to_numpy(), rolling.to_numpy(), color='tab:green', alpha=0.4, label='Rolling mean returns')
    ax_r.set_ylabel('Rolling mean returns')
    ax_r.set_ylim(rolling.min() * 1.5, rolling.max() * 1.5) if not rolling.empty else None

    # Drawdown
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    ax_dd.fill_between(drawdown.index.to_numpy(), drawdown.to_numpy(), 0, where=drawdown.to_numpy() < 0, color='tab:red')
    ax_dd.set_title('Drawdown')
    ax_dd.set_ylabel('Drawdown')
    ax_dd.grid(alpha=0.3)

    # Returns histogram
    if trade_history_path and trade_history_path.exists():
        trade_history = pd.read_csv(trade_history_path)
        if 'net_pnl' in trade_history.columns and 'entry_price' in trade_history.columns and 'size' in trade_history.columns:
            trade_returns = trade_history['net_pnl'] / (trade_history['entry_price'] * trade_history['size'])
            ax_hist.hist(trade_returns.to_numpy(), bins=60, color='tab:gray', alpha=0.9)
        else:
            ax_hist.hist(returns.to_numpy(), bins=60, color='tab:gray', alpha=0.9)
    else:
        ax_hist.hist(returns.to_numpy(), bins=60, color='tab:gray', alpha=0.9)
    ax_hist.set_title('Returns Distribution')
    ax_hist.set_xlabel('Returns')
    ax_hist.grid(alpha=0.2)
 
    # Legends
    handles, labels = ax_eq.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_eq.legend(by_label.values(), by_label.keys(), loc='upper left')
    ax_r.legend(loc='upper right')

    # save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f'Wrote plot to: {out_path}')


if __name__ == '__main__':
    in_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / 'state' / 'equity_curve.csv'
    out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent.parent / 'state' / 'backtest_plot.png'
    trade_history_csv = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).parent.parent / 'state' / 'trade_history.csv'

    if not in_csv.exists():
        print(f'Input CSV not found: {in_csv}')
        sys.exit(2)

    equity = load_equity(in_csv)
    plot_equity_diagnostics(equity, out_png, trade_history_csv)
