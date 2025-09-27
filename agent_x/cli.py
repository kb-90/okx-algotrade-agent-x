import sys
import json
from dataclasses import asdict
from .config import load_config
from .agent import Agent
from .api_client import OKXAPIClient
from .utils import clean_old_params
import subprocess
import sys
from pathlib import Path

def main():
    """Main entry point for the CLI."""
    cfg = load_config()

    # Clean old incompatible parameter files
    clean_old_params(cfg["paths"]["best_params"])

    is_demo = cfg.get("runtime", {}).get("demo", True)
    api_client = OKXAPIClient(cfg) if is_demo else None
    agent = Agent(cfg, api_client)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"

    if cmd == "train":
        print("Starting training and backtesting process...")
        metrics = agent.run_once()
        print("--- Results ---")
        print(json.dumps({"metrics": metrics, "params": asdict(agent.params)}, indent=2))
        # Attempt to generate plot after backtest/train completes
        try:
            script_path = Path(__file__).parent.parent / 'scripts' / 'plot_backtest.py'
            if script_path.exists():
                print(f"Generating backtest plot: {script_path}")
                # Pass the equity curve path to the plotting script
                equity_curve_path = Path(cfg["paths"]["equity_curve"])
                plot_output_path = Path(cfg["paths"]["state_dir"]) / "backtest_plot.png"
                trade_history_path = Path(cfg["paths"]["trade_history"])
                subprocess.run([sys.executable, str(script_path), str(equity_curve_path), str(plot_output_path), str(trade_history_path)], check=False)
            else:
                print("Plot script not found; skipping plot generation.")
        except Exception as e:
            print(f"Plot generation failed: {e}")
    elif cmd=="live":
        print("Starting live trading loop...")
        # Generate last backtest plot (if any) before entering live loop
        try:
            script_path = Path(__file__).parent.parent / 'scripts' / 'plot_backtest.py'
            if script_path.exists():
                print(f"Generating backtest plot before live mode: {script_path}")
                equity_curve_path = Path(cfg["paths"]["equity_curve"])
                plot_output_path = Path(cfg["paths"]["state_dir"]) / "backtest_plot.png"
                trade_history_path = Path(cfg["paths"]["trade_history"])
                subprocess.run([sys.executable, str(script_path), str(equity_curve_path), str(plot_output_path), str(trade_history_path)], check=False)
        except Exception as e:
            print(f"Plot generation failed: {e}")

        agent.live_loop()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python -m agent_x.cli [train|live]")

if __name__ == "__main__":
    main()
