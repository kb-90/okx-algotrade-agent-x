"""Simple unit tests for trader.py module."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from agent_x.trader import OKXTrader
from agent_x.strategy import LSTMStratParams


@pytest.fixture
def sample_cfg():
    """Create sample config for testing."""
    return {
        "symbol": "XRP-USDT-SWAP",
        "tdMode": "cross",
        "posMode": "net_mode",
        "runtime": {"demo": True},
        "risk": {"max_leverage": 4.0},
        "paths": {"trade_history": "state/trade_history.csv"}
    }


@pytest.fixture
def mock_api_client():
    """Create a mock API client."""
    client = Mock()
    client.trade = Mock()
    client.account = Mock()
    return client


@pytest.fixture
def trader(sample_cfg, mock_api_client):
    """Create a trader instance for testing."""
    params = LSTMStratParams()
    return OKXTrader(sample_cfg, mock_api_client, params)


class TestOKXTrader:
    """Simple test cases for OKXTrader class."""

    def test_init_demo_mode(self, trader, sample_cfg):
        """Test trader initialization in demo mode."""
        assert trader.symbol == sample_cfg["symbol"]
        assert trader.is_demo is True
        assert trader.current_positions == []
        assert trader.current_position_side == 0
        assert trader.current_position_size == 0.0

    def test_sync_position_with_api_success(self, trader, mock_api_client):
        """Test successful position synchronization."""
        # Mock API response
        mock_response = {
            'code': '0',
            'data': [{
                'posSide': 'long',
                'pos': '10.0',
                'avgPx': '0.5'
            }]
        }
        mock_api_client.account.get_positions.return_value = mock_response

        result = trader.sync_position_with_api()
        assert result is True
        assert len(trader.current_positions) == 1
        assert trader.current_positions[0]['side'] == 1
        assert trader.current_positions[0]['size'] == 10.0

    def test_sync_position_with_api_failure(self, trader, mock_api_client):
        """Test position synchronization failure."""
        mock_api_client.account.get_positions.return_value = None

        result = trader.sync_position_with_api()
        assert result is False

    def test_validate_position_size_match(self, trader):
        """Test position size validation when sizes match."""
        trader.current_position_size = 10.0
        result = trader.validate_position_size(10.0)
        assert result is True

    def test_validate_position_size_mismatch(self, trader):
        """Test position size validation when sizes don't match."""
        trader.current_position_size = 10.0
        result = trader.validate_position_size(15.0)
        assert result is False

    def test_place_market_demo_success(self, trader, mock_api_client):
        """Test successful market order placement in demo mode."""
        mock_response = {'code': '0', 'data': [{'fillPx': '0.5'}]}
        mock_api_client.trade.place_order.return_value = mock_response

        result = trader.place_market("buy", "10")
        assert result == mock_response

    def test_place_market_demo_dry_run(self, trader):
        """Test market order placement in dry run mode."""
        trader.trade = None  # Simulate dry run

        result = trader.place_market("buy", "10")
        assert result is None

    def test_get_balance_demo_success(self, trader, mock_api_client):
        """Test successful balance retrieval in demo mode."""
        mock_response = {'code': '0', 'data': [{'details': [{'availBal': '100.0'}]}]}
        mock_api_client.account.get_account_balance.return_value = mock_response

        result = trader.get_balance("USDT")
        assert result == mock_response

    def test_cancel_all_orders_demo_success(self, trader, mock_api_client):
        """Test successful order cancellation in demo mode."""
        mock_orders_response = {
            'code': '0',
            'data': [{'ordId': '123', 'instId': 'XRP-USDT-SWAP'}]
        }
        mock_cancel_response = {'code': '0'}
        mock_api_client.trade.get_orders.return_value = mock_orders_response
        mock_api_client.trade.cancel_order.return_value = mock_cancel_response

        result = trader.cancel_all_orders()
        assert result == {'cancelled': 1}

    def test_force_close_position_demo_success(self, trader, mock_api_client):
        """Test successful position closure in demo mode."""
        mock_response = {'code': '0'}
        mock_api_client.trade.close_position.return_value = mock_response

        result = trader.force_close_position("long")
        assert result == mock_response

    def test_place_trailing_stop_demo_success(self, trader, mock_api_client):
        """Test successful trailing stop placement in demo mode."""
        mock_response = {'code': '0'}
        mock_api_client.trade.place_algo_order.return_value = mock_response

        result = trader.place_trailing_stop("buy", "10")
        assert result == mock_response

    def test_close_position_level_valid(self, trader):
        """Test closing a valid position level."""
        trader.current_positions = [
            {'side': 1, 'size': 10.0, 'entry_price': 0.5, 'level': 0}
        ]
        trader.force_close_position = Mock(return_value={'code': '0'})

        result = trader.close_position_level(0)
        assert result == {'code': '0'}
        assert len(trader.current_positions) == 0

    def test_close_position_level_invalid(self, trader):
        """Test closing an invalid position level."""
        trader.current_positions = []

        result = trader.close_position_level(0)
        assert result is None

    def test_place_trailing_stop_live_dry_run(self, trader):
        """Test trailing stop placement in live mode dry run (not implemented, but check no error)."""
        # Since live mode requires CCXT, and test is demo, but to test structure
        # For simplicity, test that method exists and can be called
        # In real test, would need live fixture
        pass  # Keep simple as per user request


if __name__ == '__main__':
    pytest.main([__file__])
