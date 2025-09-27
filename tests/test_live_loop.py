import unittest
from unittest.mock import MagicMock, patch
import queue
import pandas as pd
from datetime import datetime, timezone, timedelta
from agent_x.agent import Agent
from agent_x.strategy import LSTMStrategy, LSTMStratParams
from agent_x.risk import RiskManager, RiskConfig

class TestLiveLoop(unittest.TestCase):
    def setUp(self):
        # Setup a minimal config dict (fix fees to float)
        self.cfg = {
            "symbol": "ADA-USDT-SWAP",
            "timeframe": "3m",
            "history_bars": 100,
            "data_source": "okx",
            "runtime": {
                "reoptimize_hours": 24,
                "base_url": "https://www.okx.com",
                "base_url_demo": "https://www.okx.com",
                "ws_private": "wss://ws.okx.com:8443/ws/v5/private",
                "ws_public": "wss://ws.okx.com:8443/ws/v5/business",
                "ws_private_demo": "wss://wspap.okx.com:8443/ws/v5/private?brokerId=9999",
                "ws_public_demo": "wss://wspap.okx.com:8443/ws/v5/business",
                "demo": True
            },
            "paths": {
                "state_dir": "state",
                "model_path": "state/lstm_model.keras",
                "best_params": "state/best_params.json",
                "trade_history": "state/trade_history.csv"
            },
            "risk": {
                "risk_per_trade": 0.1,
                "leverage": 3,
                "max_daily_loss": 0.2,
                "max_exposure": 0.9,
                "backtest_equity": 50
            },
            "lstm_params": {},
            "fees": 0.0005,  # Fixed: Use float instead of dict
            "api_key_demo": "test_key",
            "api_secret_demo": "test_secret",
            "passphrase_demo": "test_passphrase"
        }

        # Mock the entire Agent to avoid initialization issues
        self.agent = MagicMock()
        self.agent.cfg = self.cfg
        self.agent.model = MagicMock()
        self.agent.model.sequence_length = 10
        self.agent.model.predict_sequence = MagicMock(return_value=[0.05]*self.cfg["history_bars"])
        self.agent.params = LSTMStratParams()
        self.agent.trader = MagicMock()
        self.agent.trader.current_position_side = 0
        self.agent.trader.current_position_size = 0.0
        self.agent.trader.place_market = MagicMock(return_value={"data": [{"fillPx": "0.873"}]})
        self.agent.trader.get_balance = MagicMock(return_value={"code": "0", "data": [{"details": [{"availBal": "1000"}]}]})

        # Mock risk manager
        self.agent.risk = MagicMock()
        self.agent.risk.get_position_size = MagicMock(return_value=10.0)  # Float

        # Mock data
        self.agent.data = MagicMock()
        self.agent.data.fetch_history = MagicMock(return_value=pd.DataFrame({
            "timestamp": pd.date_range(end=datetime.now(timezone.utc), periods=self.cfg["history_bars"], freq="T"),
            "open": [0.873]*self.cfg["history_bars"],
            "high": [0.875]*self.cfg["history_bars"],
            "low": [0.871]*self.cfg["history_bars"],
            "close": [0.874]*self.cfg["history_bars"],
            "volume": [500000]*self.cfg["history_bars"],
        }))

        self.agent._last_reopt = datetime.now(timezone.utc) - timedelta(hours=2)
        self.agent.walk_forward_optimize = MagicMock(return_value=self.agent.params)
        self.agent.api_client = MagicMock()
        self.agent.api_client.market.get_instrument_details = MagicMock(return_value={"code": "0", "data": [{"lotSz": "0.1", "ctVal": "10"}]})
        # Mock the strategy
        self.agent.strategy = MagicMock()
        # Mock generate_signals to return tuple for scaling support
        main_signals = pd.Series([1] * self.cfg["history_bars"])
        scaling_signals = [pd.Series([0] * self.cfg["history_bars"]) for _ in range(2)]
        exit_signals = [pd.Series([0] * self.cfg["history_bars"]) for _ in range(3)]
        self.agent.strategy.generate_signals = MagicMock(return_value=(main_signals, scaling_signals, exit_signals))

    @patch("agent_x.agent.LSTMStrategy")
    @patch("agent_x.agent.RiskManager")
    @patch("agent_x.api_client.OKXWebSocketClient")
    def test_live_loop_process_message_and_reoptimize(self, MockWebSocketClient, MockRiskManager, MockLSTMStrategy):
        # Setup mock WebSocket client
        mock_ws = MockWebSocketClient.return_value
        mock_ws.message_queue = queue.Queue()
        mock_ws.wait_for_ready.return_value = True
        mock_ws.start.return_value = None

        # Mock risk manager
        mock_risk = MockRiskManager.return_value
        mock_risk.get_position_size.return_value = 10.0  # Increased for realistic sizing

        # Mock strategy
        mock_strategy = MockLSTMStrategy.return_value
        # Mock generate_signals to return tuple for scaling support
        main_signals = pd.Series([1] * self.cfg["history_bars"])
        scaling_signals = [pd.Series([0] * self.cfg["history_bars"]) for _ in range(2)]
        exit_signals = [pd.Series([0] * self.cfg["history_bars"]) for _ in range(3)]
        mock_strategy.generate_signals.return_value = (main_signals, scaling_signals, exit_signals)
        # Mock _indicator_signals to return a series with positive value for long signal
        mock_strategy._indicator_signals.return_value = pd.Series([0.5])

        # Put a fake candle message in the queue (OKX format: array)
        candle_msg = {
            "arg": {"channel": "candle3m", "instId": "ADA-USDT-SWAP"},
            "data": [[
                int(datetime.now(timezone.utc).timestamp() * 1000),
                "0.873",
                "0.875",
                "0.871",
                "0.874",
                "500000"
            ]]
        }
        mock_ws.message_queue.put(candle_msg)

        # Set mock credentials for demo mode
        import os
        os.environ["YOUR_API_KEY"] = "test_key"
        os.environ["YOUR_SECRET_KEY"] = "test_secret"
        os.environ["YOUR_PASSPHRASE"] = "test_passphrase"

        # Create a real agent for testing
        real_agent = Agent(self.cfg)
        real_agent.model = MagicMock()
        real_agent.model.sequence_length = 10
        real_agent.model.predict = MagicMock(return_value=0.88)  # Fixed: Absolute > close * (1 + threshold) for positive relative
        real_agent.model.predict_sequence = MagicMock(return_value=[0.88]*self.cfg["history_bars"])
        real_agent.params = LSTMStratParams()
        real_agent.strategy = MagicMock()
        real_agent.strategy._indicator_signals.return_value = pd.Series([0.5])
        real_agent.trader = MagicMock()
        real_agent.trader.current_position_side = 0
        real_agent.trader.current_position_size = 0.0
        real_agent.trader.place_market = MagicMock(return_value={"code": "0", "data": [{"fillPx": "0.873"}]})
        real_agent.trader.get_balance = MagicMock(return_value={"code": "0", "data": [{"details": [{"availBal": "1000"}]}]})
        real_agent.trader.get_position = MagicMock(return_value={"code": "0", "data": []})
        real_agent.trader.cancel_all_orders = MagicMock(return_value={"cancelled": 0})
        real_agent.trader.trade = True  # Enable trading
        real_agent._last_reopt = datetime.now(timezone.utc) - timedelta(hours=2)
        real_agent.walk_forward_optimize = MagicMock(return_value=real_agent.params)
        real_agent.api_client = MagicMock()
        real_agent.api_client.market.get_instrument_details = MagicMock(return_value={"code": "0", "data": [{"lotSz": "0.1", "ctVal": "10"}]})
        # Mock the data fetch_history method
        real_agent.data.fetch_history = MagicMock(return_value=pd.DataFrame({
            "timestamp": pd.date_range(end=datetime.now(timezone.utc), periods=self.cfg["history_bars"], freq="min"),
            "open": [0.873]*self.cfg["history_bars"],
            "high": [0.875]*self.cfg["history_bars"],
            "low": [0.871]*self.cfg["history_bars"],
            "close": [0.874]*self.cfg["history_bars"],
            "volume": [500000]*self.cfg["history_bars"],
        }))
        # Mock the API client availability
        import agent_x.agent as agent_module
        agent_module.API_CLIENT_AVAILABLE = True

        # Run live_loop in a limited way: patch time.sleep to raise to break loop
        with patch("agent_x.agent.RiskManager") as MockRisk:
            mock_risk_instance = MockRisk.return_value
            mock_risk_instance.get_position_size.return_value = 10.0
            with patch("time.sleep", side_effect=Exception("StopLoop")):
                try:
                    real_agent.live_loop()
                except Exception as e:
                    self.assertEqual(str(e), "StopLoop")

        # Check that place_market was called to enter long position
        real_agent.trader.place_market.assert_called()

        # Check that walk_forward_optimize was called for re-optimization
        real_agent.walk_forward_optimize.assert_called()

if __name__ == "__main__":
    unittest.main()
