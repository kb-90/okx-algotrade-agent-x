"""Unit tests for strategy.py module."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from agent_x.strategy import LSTMStrategy, LSTMStratParams
from agent_x.risk import RiskManager
from agent_x.model import LSTMModel


@pytest.fixture
def sample_params():
    """Create sample strategy parameters for testing."""
    return LSTMStratParams(
        prediction_threshold=0.002,
        indicator_weights={'rsi': 0.4, 'macd': 0.36, 'bb': 0.24},
        exit_mode='intelligent',
        scaling_thresholds=[0.01, 0.02],
        partial_exit_thresholds=[0.005, 0.01, 0.015]
    )


@pytest.fixture
def mock_risk_manager():
    """Create a mock risk manager."""
    rm = Mock(spec=RiskManager)
    rm.config = Mock()
    rm.config.max_open_orders = 3
    return rm


@pytest.fixture
def mock_model():
    """Create a mock LSTM model."""
    model = Mock(spec=LSTMModel)
    model.predict_sequence.return_value = np.array([0.001, 0.002, -0.001])
    return model


@pytest.fixture
def strategy(sample_params, mock_risk_manager, mock_model):
    """Create a strategy instance for testing."""
    return LSTMStrategy(sample_params, mock_risk_manager, mock_model)


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='1H')
    np.random.seed(42)
    close_prices = 50000 + np.random.randn(100) * 1000

    df = pd.DataFrame({
        'close': close_prices,
        'rsi': np.random.uniform(20, 80, 100),
        'macd': np.random.randn(100) * 0.001,
        'macd_signal': np.random.randn(100) * 0.001,
        'bb_upper': close_prices + np.random.uniform(500, 1000, 100),
        'bb_lower': close_prices - np.random.uniform(500, 1000, 100),
        'prediction': np.random.randn(100) * 0.005,
        'atr': np.random.uniform(100, 500, 100)
    }, index=dates)

    return df


class TestLSTMStrategy:
    """Test cases for LSTMStrategy class."""

    def test_init(self, strategy, sample_params):
        """Test strategy initialization."""
        assert strategy.params == sample_params
        assert strategy.fees == 0.0005

    def test_require_columns_success(self, strategy, sample_df):
        """Test _require_columns with all required columns present."""
        strategy._require_columns(sample_df, ['close', 'rsi'])
        # Should not raise exception

    def test_require_columns_missing(self, strategy, sample_df):
        """Test _require_columns with missing columns."""
        with pytest.raises(ValueError, match="Missing required feature columns"):
            strategy._require_columns(sample_df, ['close', 'missing_column'])

    def test_indicator_signals(self, strategy, sample_df):
        """Test indicator signal computation."""
        signals = strategy._indicator_signals(sample_df)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_df)
        assert signals.index.equals(sample_df.index)
        # Signals should be between -1 and 1 (weighted sum)
        assert signals.min() >= -1
        assert signals.max() <= 1

    def test_handle_entry_signals_long(self, strategy, sample_df):
        """Test long entry signal detection."""
        # Set up conditions for long entry
        sample_df.loc[sample_df.index[0], 'prediction'] = 0.005  # Above threshold
        sample_df.loc[sample_df.index[0], 'rsi'] = 25  # Oversold
        sample_df.loc[sample_df.index[0], 'macd'] = 0.002
        sample_df.loc[sample_df.index[0], 'macd_signal'] = 0.001
        sample_df.loc[sample_df.index[0], 'close'] = 49000
        sample_df.loc[sample_df.index[0], 'bb_lower'] = 49500

        signal = strategy._handle_entry_signals(sample_df, 0, 50000.0)
        assert signal == 1  # Long signal

    def test_handle_entry_signals_short(self, strategy, sample_df):
        """Test short entry signal detection."""
        # Set up conditions for short entry
        sample_df.loc[sample_df.index[0], 'prediction'] = -0.005  # Below threshold
        sample_df.loc[sample_df.index[0], 'rsi'] = 75  # Overbought
        sample_df.loc[sample_df.index[0], 'macd'] = -0.002
        sample_df.loc[sample_df.index[0], 'macd_signal'] = -0.001
        sample_df.loc[sample_df.index[0], 'close'] = 51000
        sample_df.loc[sample_df.index[0], 'bb_upper'] = 50500

        signal = strategy._handle_entry_signals(sample_df, 0, 50000.0)
        assert signal == -1  # Short signal

    def test_handle_entry_signals_no_signal(self, strategy, sample_df):
        """Test no entry signal when conditions not met."""
        signal = strategy._handle_entry_signals(sample_df, 0, 50000.0)
        assert signal == 0

    def test_handle_exit_signals_mechanical_atr(self, strategy, sample_df):
        """Test mechanical exit with ATR stop."""
        strategy.params.exit_mode = 'mechanical'
        strategy.params.atr_mult_sl = 2.0

        # Set up ATR-based exit condition
        sample_df.loc[sample_df.index[0], 'atr'] = 1000  # Large ATR
        entry_price = 50000.0
        current_price = 47900.0  # Significant drop beyond ATR stop

        exit_signal = strategy._handle_exit_signals(
            sample_df, 0, current_price, 1, entry_price, 0.0
        )
        assert exit_signal is True

    def test_handle_exit_signals_lstm_disagreement(self, strategy, sample_df):
        """Test exit due to LSTM prediction disagreement."""
        strategy.params.exit_mode = 'intelligent'
        strategy.params.lstm_disagreement_pct = 0.01

        # Long position with negative prediction
        sample_df.loc[sample_df.index[0], 'prediction'] = -0.02

        exit_signal = strategy._handle_exit_signals(
            sample_df, 0, 50000.0, 1, 50000.0, 0.0
        )
        assert exit_signal is True

    def test_handle_exit_signals_equity_based_small_account(self, strategy, sample_df):
        """Test equity-based exits for small accounts."""
        exit_signal = strategy._handle_exit_signals(
            sample_df, 0, 52525.0, 1, 50000.0, 0.0, equity=25.0  # Small account
        )
        assert exit_signal is True  # Should exit on 5% profit

    def test_handle_exit_signals_equity_based_large_account(self, strategy, sample_df):
        """Test equity-based exits for large accounts."""
        exit_signal = strategy._handle_exit_signals(
            sample_df, 0, 53762.5, 1, 50000.0, 0.0, equity=1000.0  # Large account
        )
        assert exit_signal is True  # Should exit on 7.5% profit

    def test_generate_signals_empty_df(self, strategy):
        """Test generate_signals with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = strategy.generate_signals(empty_df)

        assert len(result) == 3
        assert isinstance(result[0], pd.Series)
        assert isinstance(result[1], list)
        assert isinstance(result[2], list)
        assert all(sig.empty for sig in result[1]) if result[1] else True
        assert all(sig.empty for sig in result[2]) if result[2] else True

    def test_generate_signals_with_predictions(self, strategy, sample_df):
        """Test signal generation with prediction data."""
        result = strategy.generate_signals(sample_df)

        main_signals, scaling_signals, exit_signals = result

        assert isinstance(main_signals, pd.Series)
        assert isinstance(scaling_signals, list)
        assert isinstance(exit_signals, list)
        assert len(scaling_signals) == 2  # max_open_orders - 1
        assert len(exit_signals) == 3  # max_open_orders

    def test_generate_signals_without_predictions(self, strategy, sample_df):
        """Test signal generation without prediction data."""
        df_no_pred = sample_df.drop(columns=['prediction'])
        result = strategy.generate_signals(df_no_pred)

        main_signals, _, _ = result
        assert isinstance(main_signals, pd.Series)
        assert len(main_signals) == len(df_no_pred)

    def test_trailing_stop_initial_loss_exit(self, strategy, sample_df):
        """Test initial trailing stop loss exit when loss exceeds threshold."""
        strategy.params.exit_mode = 'mechanical'
        strategy.params.initial_trail_pct = 0.05  # 5% initial trail

        # Simulate a position that has lost more than initial trail
        entry_price = 50000.0
        current_price = 47500.0  # 5% loss
        pnl = (current_price - entry_price) / entry_price  # -0.05

        exit_signal = strategy._handle_exit_signals(
            sample_df, 0, current_price, 1, entry_price, 0.0
        )
        assert exit_signal is True  # Should exit due to initial trail stop

    def test_trailing_stop_profit_trigger_activates_tighter_trail(self, strategy, sample_df):
        """Test that profit trigger activates tighter trailing stop."""
        strategy.params.exit_mode = 'mechanical'
        strategy.params.profit_trigger_pct = 0.03  # 3% profit trigger
        strategy.params.tighter_trail_pct = 0.01  # 1% tighter trail
        strategy.params.initial_trail_pct = 0.05  # 5% initial trail

        # Position has reached profit trigger (3%)
        entry_price = 50000.0
        highest_profit = 0.03  # 3% profit achieved
        current_price = 51450.0  # Current profit of 2.9% (below tighter trail threshold)
        pnl = (current_price - entry_price) / entry_price  # 0.029

        exit_signal = strategy._handle_exit_signals(
            sample_df, 0, current_price, 1, entry_price, highest_profit
        )
        assert exit_signal is False  # Should NOT exit, tighter trail not triggered yet

    def test_trailing_stop_tighter_trail_exit(self, strategy, sample_df):
        """Test exit when tighter trailing stop is triggered after profit target."""
        strategy.params.exit_mode = 'mechanical'
        strategy.params.profit_trigger_pct = 0.03  # 3% profit trigger
        strategy.params.tighter_trail_pct = 0.01  # 1% tighter trail

        # Position reached profit trigger, now test tighter trail
        entry_price = 50000.0
        highest_profit = 0.03  # Profit trigger reached
        current_price = 50950.0  # Profit dropped to 1.9% (below tighter trail threshold of 2%)
        pnl = (current_price - entry_price) / entry_price  # 0.019

        exit_signal = strategy._handle_exit_signals(
            sample_df, 0, current_price, 1, entry_price, highest_profit
        )
        assert exit_signal is True  # Should exit due to tighter trail stop

    def test_trailing_stop_initial_trail_after_small_profit(self, strategy, sample_df):
        """Test initial trailing stop when profit is below profit trigger."""
        strategy.params.exit_mode = 'mechanical'
        strategy.params.profit_trigger_pct = 0.03  # 3% profit trigger
        strategy.params.initial_trail_pct = 0.015  # 1.5% initial trail

        # Position has small profit but below profit trigger
        entry_price = 50000.0
        highest_profit = 0.02  # 2% profit (below trigger but above initial trail)
        current_price = 50950.0  # Profit dropped to 1.9% (above initial trail level)
        pnl = (current_price - entry_price) / entry_price  # 0.019

        exit_signal = strategy._handle_exit_signals(
            sample_df, 0, current_price, 1, entry_price, highest_profit
        )
        assert exit_signal is False  # Should NOT exit, current profit is above trail level

    def test_order_placement_scenario_with_trailing_stops(self, strategy, sample_df):
        """Test complete order placement scenario with trailing stop logic."""
        # Set up strategy parameters
        strategy.params.exit_mode = 'mechanical'
        strategy.params.profit_trigger_pct = 0.03
        strategy.params.tighter_trail_pct = 0.01
        strategy.params.initial_trail_pct = 0.05

        # Simulate price movement over time
        entry_price = 50000.0
        prices = [
            50000.0,  # Entry
            51500.0,  # +3% profit (triggers tighter trail)
            51400.0,  # +2.8% profit (still above tighter trail threshold of 2%)
            51200.0,  # +2.4% profit (still above tighter trail)
            50950.0,  # +1.9% profit (below tighter trail - should exit)
        ]

        highest_profit = 0.0
        position_active = True

        for i, current_price in enumerate(prices):
            if not position_active:
                break

            pnl = (current_price - entry_price) / entry_price
            highest_profit = max(highest_profit, pnl)

            exit_signal = strategy._handle_exit_signals(
                sample_df, i, current_price, 1, entry_price, highest_profit
            )

            if exit_signal:
                position_active = False
                if i == 4:  # Should exit at the last price point
                    assert True  # Correct exit timing
                else:
                    assert False, f"Unexpected exit at step {i}"
            else:
                if i < 4:  # Should not exit until last step
                    assert True
                else:
                    assert False, "Should have exited at last step"


if __name__ == '__main__':
    pytest.main([__file__])
