import os
from .utils import logger
from . import api_client
import csv
from pathlib import Path
from datetime import datetime
import threading
from typing import Optional, Dict, Any
 
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

# Use a lock to prevent race conditions when writing to the trade history file
_trade_history_lock = threading.Lock()

class OKXTrader:
    def __init__(self, cfg, api_client: api_client.OKXAPIClient = None, params=None):
        self.cfg = cfg
        if params:
            self.params = params
        else:
            from .strategy import LSTMStratParams
            self.params = LSTMStratParams()
        self.symbol = cfg["symbol"]
        self.tdMode = cfg.get("tdMode", "cross")
        self.is_demo = cfg.get("runtime", {}).get("demo", True)
        self.fees = self.cfg.get("fees", 0.0005)

        if self.is_demo:
            # Demo mode: use OKX REST API with x-simulated-trading header
            self.api_client = api_client
            if self.api_client:
                self.trade = self.api_client.trade
                self.account = self.api_client.account
                # Set position mode (only if no open positions/orders)
                pos_mode = self.cfg.get("posMode", "net_mode")
                response = self.trade.set_position_mode(pos_mode)
                if response and response.get('code') == '0':
                    logger.info(f"Position mode set to {pos_mode}")
                else:
                    logger.warning(f"TradeAPI Error: Setting failed. Cancel any open orders, close positions, and stop trading bots first. (Code: {response.get('code', 'Unknown')}) for path /api/v5/account/set-position-mode")
                    logger.warning(f"Failed to set position mode: {response.get('msg', 'Unknown error')}")
                # Set leverage from config for cross margin
                leverage = self.cfg.get("risk", {}).get("max_leverage", 3)
                if self.tdMode == 'cross':
                    if pos_mode == "long_short_mode":
                        # Set leverage for both long and short
                        response = self.trade.set_leverage(self.symbol, leverage, 'cross')
                        if response and response.get('code') == '0':
                            logger.info(f"Set leverage to {leverage}x for both long and short in hedge mode")
                        else:
                            logger.warning("Failed to set leverage for hedge mode")
                    else:
                        response = self.trade.set_leverage(self.symbol, leverage, 'cross')
                        if response and response.get('code') == '0':
                            logger.info(f"Set leverage to {leverage}x")
                        else:
                            logger.warning("Failed to set leverage")
            else:
                self.trade = None
                self.account = None
                logger.warning("Trade API client not available. Demo trading is disabled.")
        else:

            # Live mode: use CCXT
            if not CCXT_AVAILABLE:
                raise ImportError("CCXT library is required for live trading.")
            self.api_client = None
            self.exchange = ccxt.okx({
                'apiKey': os.getenv("YOUR_API_KEY"),
                'secret': os.getenv("YOUR_SECRET_KEY"),
                'password': os.getenv("YOUR_PASSPHRASE"),
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'},
            })
            # Ensure no simulated header for live
            headers = self.exchange.headers if hasattr(self.exchange, "headers") and isinstance(self.exchange.headers, dict) else {}
            headers.pop("x-simulated-trading", None)
            self.exchange.headers = headers
            self.exchange.load_markets()
            # Set leverage
            leverage = self.cfg["risk"]["max_leverage"]
            try:
                self.exchange.set_leverage(leverage, self.symbol, {"mgnMode": self.tdMode})
                logger.info(f"Leverage set to {leverage}x for {self.symbol}")
            except Exception as e:
                logger.warning(f"Could not set leverage via CCXT: {e}")
            self.trade = None  # Not used in live mode
            self.account = None  # Not used in live mode

        # Initialize position tracking
        self.current_positions = []  # List of dicts: {'side': int, 'size': float, 'entry_price': float, 'level': int}
        self.current_position_side = 0  # 0: flat, 1: long, -1: short (for backward compatibility)
        self.current_position_size = 0.0  # Total position size (for backward compatibility)

    def sync_position_with_api(self) -> bool:
        """Synchronize local position tracking with API data."""
        try:
            position_response = self.get_position()
            if position_response and position_response.get('code') == '0' and position_response.get('data'):
                positions = position_response['data']
                self.current_positions = []
                for pos_data in positions:
                    if pos_data:
                        pos_side_str = pos_data.get('posSide', 'net')
                        pos_size = float(pos_data.get('pos', 0))
                        entry_price = float(pos_data.get('avgPx', 0)) if 'avgPx' in pos_data else 0.0
                        if pos_side_str == 'long':
                            side = 1
                        elif pos_side_str == 'short':
                            side = -1
                        else:
                            side = 0
                        if side != 0 and pos_size > 0:
                            level = len(self.current_positions)
                            self.current_positions.append({
                                'side': side,
                                'size': pos_size,
                                'entry_price': entry_price,
                                'level': level
                            })

                # Update backward compatible single position side and size
                if self.current_positions:
                    total_size = sum(p['size'] for p in self.current_positions)
                    avg_side = max(set(p['side'] for p in self.current_positions), key=lambda s: sum(p['size'] for p in self.current_positions if p['side'] == s))
                    self.current_position_side = avg_side
                    self.current_position_size = total_size
                else:
                    self.current_position_side = 0
                    self.current_position_size = 0.0

                logger.info(f"Position synchronized: {len(self.current_positions)} positions, total size {self.current_position_size}")
                return True
            else:
                logger.warning("Failed to sync position with API")
                return False
        except Exception as e:
            logger.error(f"Error syncing position: {e}")
            return False

    def validate_position_size(self, expected_size: float, tolerance: float = 0.01) -> bool:
        """Validate that current position size matches expected size within tolerance."""
        if abs(self.current_position_size - expected_size) > tolerance:
            logger.warning(f"Position size mismatch: expected {expected_size}, actual {self.current_position_size}")
            # Try to sync with API
            self.sync_position_with_api()
            return abs(self.current_position_size - expected_size) <= tolerance
        return True

    def get_position(self, instId: Optional[str] = None) -> Optional[Dict[str, Any]]:
        symbol = instId or self.symbol
        if self.is_demo:
            if not self.account:
                logger.warning("Account API client not available for demo mode.")
                return None
            response = self.account.get_positions(instId=symbol)
            return response
        else:
            try:
                if symbol == self.symbol:
                    positions = self.exchange.fetch_positions([self.symbol])
                    # CCXT returns a list of positions, we need to find the one for our symbol
                    for position in positions:
                        if position['info']['instId'] == self.symbol:
                            return {
                                'code': '0',
                                'data': [position['info']]
                            }
                    return {'code': '0', 'data': []} # No position found
                else:
                    # For other symbols, CCXT fetch_positions() gets all
                    positions = self.exchange.fetch_positions()
                    filtered = [p for p in positions if p['info']['instId'] == symbol]
                    return {
                        'code': '0',
                        'data': [p['info'] for p in filtered]
                    }
            except Exception as e:
                logger.error(f"CCXT position fetch failed: {e}")
                return None

    def place_scaling_order(self, side: str, sz: str, position_level: int) -> Optional[Any]:
        """Place a scaling order for additional position levels."""
        return self.place_market(side, sz)

    def close_position_level(self, position_level: int) -> Optional[Any]:
        """Close a specific position level."""
        if position_level < len(self.current_positions):
            pos = self.current_positions[position_level]
            side_to_close = 'long' if pos['side'] == 1 else 'short'
            sz = str(pos['size'])
            response = self.force_close_position(side_to_close)
            if response and response.get('code') == '0':
                logger.info(f"Closed position level {position_level}: {side_to_close} {sz}")
                # Remove the position from tracking
                self.current_positions.pop(position_level)
                # Update backward compatible fields
                if self.current_positions:
                    self.current_position_size = sum(p['size'] for p in self.current_positions)
                    self.current_position_side = self.current_positions[0]['side']  # Assume all same side
                else:
                    self.current_position_side = 0
                    self.current_position_size = 0.0
            return response
        return None

    def place_market(self, side: str, sz: str) -> Optional[Any]:
        """Place a market order for the given side and size."""
        logger.info(f"Placing market order ({'DEMO' if self.is_demo else 'LIVE'}): {side} {sz} {self.symbol}")
        if self.is_demo:
            response = self._place_market_demo(side, sz)
        else:
            response = self._place_market_live(side, sz)
        self._log_order_placement(side, sz, response, is_live=not self.is_demo)
        return response

    def _place_market_demo(self, side: str, sz: str) -> Optional[Any]:
        """Place market order in demo mode using OKX REST API."""
        if not self.trade:
            logger.warning(f"DRY RUN: Would place {side} {sz} {self.symbol}")
            self._append_trade_record({'ts': datetime.utcnow(), 'event': 'order_placed', 'side': side, 'size': sz, 'price': None, 'pnl': None, 'fees': None})
            return None

        body = self._build_order_body(side, sz, "market")
        response = self.trade.place_order(body)
        logger.info(f"Order placement response: {response}")
        return response

    def _place_market_live(self, side: str, sz: str) -> Optional[Any]:
        """Place market order in live mode using CCXT."""
        try:
            params = {"tdMode": self.tdMode, "posSide": "long" if side == "buy" else "short"}
            response = self.exchange.create_order(self.symbol, "market", side, float(sz), None, params)
            logger.info(f"Order placement response: {response}")
            return response
        except Exception as e:
            logger.error(f"CCXT order placement failed: {e}")
            return None

    def _build_order_body(self, side: str, sz: str, ord_type: str) -> dict:
        """Build the order body for OKX API."""
        body = {
            "instId": self.symbol,
            "tdMode": self.tdMode,
            "side": side,
            "ordType": ord_type,
            "sz": str(sz),
        }
        pos_mode = self.cfg.get("posMode", "net_mode")
        if pos_mode == "long_short_mode":
            body["posSide"] = 'long' if side == 'buy' else 'short'
        return body

    def _log_order_placement(self, side: str, sz: str, response, is_live: bool = False) -> None:
        """Log order placement and append to trade history."""
        try:
            if is_live:
                px = response.get('price') if response else None
            else:
                px = None
                if response and response.get('data') and isinstance(response['data'], list) and len(response['data']) > 0:
                    px = response['data'][0].get('fillPx')

            # Calculate fees
            contract_size = self.cfg.get("contract_size", 1)
            amount = float(sz) * contract_size * (float(px) if px else 1.0)
            fees_paid_calc = self.fees * amount

            self._append_trade_record({
                'ts': datetime.utcnow(),
                'event': 'order_placed',
                'side': side,
                'size': sz,
                'price': px,
                'pnl': None,
                'fees': fees_paid_calc
            })
        except Exception as e:
            logger.debug(f"Could not append trade record: {e}")

    def get_balance(self, ccy: str = 'USDT') -> Optional[Any]:
        if self.is_demo:
            # Demo mode: use OKX REST API
            if not self.account:
                logger.warning("Account API client not available for demo mode.")
                return None
            response = self.account.get_account_balance(ccy=ccy)
            return response
        else:
            # Live mode: use CCXT
            try:
                balance = self.exchange.fetch_balance()
                return {
                    'code': '0',
                    'data': [{
                        'details': [{
                            'availBal': str(balance.get(ccy, {}).get('free', 0))
                        }]
                    }]
                }
            except Exception as e:
                logger.error(f"CCXT balance fetch failed: {e}")
                return None

    def cancel_all_orders(self, instId: Optional[str] = None) -> Optional[Dict[str, int]]:
        """Cancel all pending orders for the symbol or all if instId is None."""
        if self.is_demo:
            if not self.trade:
                logger.warning("Trade API client not available for demo mode.")
                return None
            if instId is None:
                logger.info("Cancelling all pending orders for all instruments")
                orders_response = self.trade.get_orders(instId=None)
            else:
                logger.info(f"Cancelling all pending orders for {instId}")
                orders_response = self.trade.get_orders(instId=instId)
            if orders_response and orders_response.get('code') == '0' and orders_response.get('data'):
                cancelled_count = 0
                for order in orders_response['data']:
                    ordId = order.get('ordId')
                    inst_id = order.get('instId', instId or self.symbol)
                    if ordId:
                        cancel_response = self.trade.cancel_order(inst_id, ordId=ordId)
                        if cancel_response and cancel_response.get('code') == '0':
                            cancelled_count += 1
                            logger.info(f"Cancelled order {ordId} for {inst_id}")
                        else:
                            logger.warning(f"Failed to cancel order {ordId} for {inst_id}")
                logger.info(f"Cancelled {cancelled_count} pending orders")
                return {'cancelled': cancelled_count}
            else:
                logger.info("No pending orders to cancel")
                return {'cancelled': 0}
        else:
            # Live mode: use CCXT
            try:
                if instId is None:
                    logger.info("Cancelling all pending orders for all instruments via CCXT")
                    orders = self.exchange.fetch_open_orders()
                else:
                    logger.info(f"Cancelling all pending orders for {instId} via CCXT")
                    orders = self.exchange.fetch_open_orders(instId)
                cancelled_count = 0
                for order in orders:
                    try:
                        self.exchange.cancel_order(order['id'], order['symbol'])
                        cancelled_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to cancel order {order['id']}: {e}")
                logger.info(f"Cancelled {cancelled_count} pending orders")
                return {'cancelled': cancelled_count}
            except Exception as e:
                logger.error(f"CCXT cancel all orders failed: {e}")
                return None

    def force_close_position(self, pos_side: str) -> Optional[Any]:
        """Force close an existing position."""
        if self.is_demo:
            if not self.trade:
                logger.warning("Trade API client not available for demo mode.")
                return None
            logger.info(f"Force closing {pos_side} position")
            response = self.trade.close_position(self.symbol, self.tdMode, pos_side)
            if response and response.get('code') == '0':
                # Log the exit
                self._append_trade_record({'ts': datetime.utcnow(), 'event': 'position_closed', 'side': pos_side, 'size': None, 'price': None, 'pnl': None, 'fees': None})
            return response
        else:
            # Live mode: use CCXT
            try:
                logger.info(f"Force closing {pos_side} position via CCXT")
                # CCXT close_position method
                response = self.exchange.close_position(self.symbol, {"posSide": pos_side})
                if response:
                    # Log the exit
                    self._append_trade_record({'ts': datetime.utcnow(), 'event': 'position_closed', 'side': pos_side, 'size': None, 'price': None, 'pnl': None, 'fees': None})
                return response
            except Exception as e:
                logger.error(f"CCXT force close failed: {e}")
                return None

    def place_trailing_stop(self, side: str, sz: str) -> Optional[Any]:
        """Place a trailing stop order."""
        logger.info(f"Placing trailing stop order ({'DEMO' if self.is_demo else 'LIVE'}): {side} {sz} {self.symbol}")
        if self.is_demo:
            return self._place_trailing_stop_demo(side, sz)
        else:
            return self._place_trailing_stop_live(side, sz)

    def _place_trailing_stop_demo(self, side: str, sz: str) -> Optional[Any]:
        """Place trailing stop order in demo mode using OKX REST API."""
        if not self.trade:
            logger.warning(f"DRY RUN: Would place trailing stop {side} {sz} {self.symbol}")
            return None

        body = {
            "instId": self.symbol,
            "tdMode": self.tdMode,
            "side": side,
            "ordType": "trailingStop",
            "sz": str(sz),
            "callbackRatio": f"{self.params.initial_trail_pct}",
        }
        try:
            response = self.trade.place_algo_order(body)
            logger.info(f"Trailing stop placement response: {response}")
            return response
        except Exception as e:
            logger.error(f"Trailing stop placement failed: {e}")
            return None

    def _place_trailing_stop_live(self, side: str, sz: str) -> Optional[Any]:
        """Place trailing stop order in live mode using CCXT."""
        try:
            # CCXT for OKX supports algo orders via create_order with ordType
            params = {
                "ordType": "trailingStop",
                "sz": str(sz),
                "callbackRatio": f"{self.params.initial_trail_pct}",
                "tdMode": self.tdMode,
                "posSide": "long" if side == "buy" else "short"
            }
            response = self.exchange.create_order(self.symbol, "market", side, float(sz), None, params)
            logger.info(f"Trailing stop placement response: {response}")
            return response
        except Exception as e:
            logger.error(f"CCXT trailing stop placement failed: {e}")
            return None

    def _append_trade_record(self, record: dict) -> None:
        """Append a trade event to the configured trade_history CSV.

        record keys expected: ts (datetime), event (str), side (str), size, price, pnl
        """
        path = self.cfg.get('paths', {}).get('trade_history')
        if not path:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        with _trade_history_lock: # Use the lock here
            write_header = not p.exists()
            with p.open('a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['ts', 'event', 'side', 'size', 'price', 'pnl', 'fees'])
                import os
from .utils import logger
from . import api_client
import csv
from pathlib import Path
from datetime import datetime
import threading
from typing import Optional, Dict, Any

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    PLAYSOUND_AVAILABLE = False

# Use a lock to prevent race conditions when writing to the trade history file
_trade_history_lock = threading.Lock()

class OKXTrader:
    def __init__(self, cfg, api_client: api_client.OKXAPIClient = None, params=None):
        self.cfg = cfg
        if params:
            self.params = params
        else:
            from .strategy import LSTMStratParams
            self.params = LSTMStratParams()
        self.symbol = cfg["symbol"]
        self.tdMode = cfg.get("tdMode", "cross")
        self.is_demo = cfg.get("runtime", {}).get("demo", True)

        if self.is_demo:
            # Demo mode: use OKX REST API with x-simulated-trading header
            self.api_client = api_client
            if self.api_client:
                self.trade = self.api_client.trade
                self.account = self.api_client.account
                # Set position mode (only if no open positions/orders)
                pos_mode = self.cfg.get("posMode", "net_mode")
                response = self.trade.set_position_mode(pos_mode)
                if response and response.get('code') == '0':
                    logger.info(f"Position mode set to {pos_mode}")
                else:
                    logger.warning(f"TradeAPI Error: Setting failed. Cancel any open orders, close positions, and stop trading bots first. (Code: {response.get('code', 'Unknown')}) for path /api/v5/account/set-position-mode")
                    logger.warning(f"Failed to set position mode: {response.get('msg', 'Unknown error')}")
                # Set leverage from config for cross margin
                leverage = self.cfg.get("risk", {}).get("leverage", 3)
                if self.tdMode == 'cross':
                    if pos_mode == "long_short_mode":
                        # Set leverage for both long and short
                        response = self.trade.set_leverage(self.symbol, leverage, 'cross')
                        if response and response.get('code') == '0':
                            logger.info(f"Set leverage to {leverage}x for both long and short in hedge mode")
                        else:
                            logger.warning("Failed to set leverage for hedge mode")
                    else:
                        response = self.trade.set_leverage(self.symbol, leverage, 'cross')
                        if response and response.get('code') == '0':
                            logger.info(f"Set leverage to {leverage}x")
                        else:
                            logger.warning("Failed to set leverage")
            else:
                self.trade = None
                self.account = None
                logger.warning("Trade API client not available. Demo trading is disabled.")
        else:

            # Live mode: use CCXT
            if not CCXT_AVAILABLE:
                raise ImportError("CCXT library is required for live trading.")
            self.api_client = None
            self.exchange = ccxt.okx({
                'apiKey': os.getenv("YOUR_API_KEY"),
                'secret': os.getenv("YOUR_SECRET_KEY"),
                'password': os.getenv("YOUR_PASSPHRASE"),
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'},
            })
            # Ensure no simulated header for live
            headers = self.exchange.headers if hasattr(self.exchange, "headers") and isinstance(self.exchange.headers, dict) else {}
            headers.pop("x-simulated-trading", None)
            self.exchange.headers = headers
            self.exchange.load_markets()
            # Set leverage
            leverage = self.cfg["risk"]["leverage"]
            try:
                self.exchange.set_leverage(leverage, self.symbol, {"mgnMode": self.tdMode})
                logger.info(f"Leverage set to {leverage}x for {self.symbol}")
            except Exception as e:
                logger.warning(f"Could not set leverage via CCXT: {e}")
            self.trade = None  # Not used in live mode
            self.account = None  # Not used in live mode

        # Initialize position tracking
        self.current_positions = []  # List of dicts: {'side': int, 'size': float, 'entry_price': float, 'level': int}
        self.current_position_side = 0  # 0: flat, 1: long, -1: short (for backward compatibility)
        self.current_position_size = 0.0  # Total position size (for backward compatibility)

    def sync_position_with_api(self) -> bool:
        """Synchronize local position tracking with API data."""
        try:
            position_response = self.get_position()
            if position_response and position_response.get('code') == '0' and position_response.get('data'):
                positions = position_response['data']
                self.current_positions = []
                for pos_data in positions:
                    if pos_data:
                        pos_side_str = pos_data.get('posSide', 'net')
                        pos_size = float(pos_data.get('pos', 0))
                        entry_price = float(pos_data.get('avgPx', 0)) if 'avgPx' in pos_data else 0.0
                        if pos_side_str == 'long':
                            side = 1
                        elif pos_side_str == 'short':
                            side = -1
                        else:
                            side = 0
                        if side != 0 and pos_size > 0:
                            level = len(self.current_positions)
                            self.current_positions.append({
                                'side': side,
                                'size': pos_size,
                                'entry_price': entry_price,
                                'level': level
                            })

                # Update backward compatible single position side and size
                if self.current_positions:
                    total_size = sum(p['size'] for p in self.current_positions)
                    avg_side = max(set(p['side'] for p in self.current_positions), key=lambda s: sum(p['size'] for p in self.current_positions if p['side'] == s))
                    self.current_position_side = avg_side
                    self.current_position_size = total_size
                else:
                    self.current_position_side = 0
                    self.current_position_size = 0.0

                logger.info(f"Position synchronized: {len(self.current_positions)} positions, total size {self.current_position_size}")
                return True
            else:
                logger.warning("Failed to sync position with API")
                return False
        except Exception as e:
            logger.error(f"Error syncing position: {e}")
            return False

    def validate_position_size(self, expected_size: float, tolerance: float = 0.01) -> bool:
        """Validate that current position size matches expected size within tolerance."""
        if abs(self.current_position_size - expected_size) > tolerance:
            logger.warning(f"Position size mismatch: expected {expected_size}, actual {self.current_position_size}")
            # Try to sync with API
            self.sync_position_with_api()
            return abs(self.current_position_size - expected_size) <= tolerance
        return True

    def get_position(self, instId: Optional[str] = None) -> Optional[Dict[str, Any]]:
        symbol = instId or self.symbol
        if self.is_demo:
            if not self.account:
                logger.warning("Account API client not available for demo mode.")
                return None
            response = self.account.get_positions(instId=symbol)
            return response
        else:
            try:
                if symbol == self.symbol:
                    positions = self.exchange.fetch_positions([self.symbol])
                    # CCXT returns a list of positions, we need to find the one for our symbol
                    for position in positions:
                        if position['info']['instId'] == self.symbol:
                            return {
                                'code': '0',
                                'data': [position['info']]
                            }
                    return {'code': '0', 'data': []} # No position found
                else:
                    # For other symbols, CCXT fetch_positions() gets all
                    positions = self.exchange.fetch_positions()
                    filtered = [p for p in positions if p['info']['instId'] == symbol]
                    return {
                        'code': '0',
                        'data': [p['info'] for p in filtered]
                    }
            except Exception as e:
                logger.error(f"CCXT position fetch failed: {e}")
                return None

    def place_scaling_order(self, side: str, sz: str, position_level: int) -> Optional[Any]:
        """Place a scaling order for additional position levels."""
        return self.place_market(side, sz)

    def close_position_level(self, position_level: int) -> Optional[Any]:
        """Close a specific position level."""
        if position_level < len(self.current_positions):
            pos = self.current_positions[position_level]
            side_to_close = 'long' if pos['side'] == 1 else 'short'
            sz = str(pos['size'])
            response = self.force_close_position(side_to_close)
            if response and response.get('code') == '0':
                logger.info(f"Closed position level {position_level}: {side_to_close} {sz}")
                # Remove the position from tracking
                self.current_positions.pop(position_level)
                # Update backward compatible fields
                if self.current_positions:
                    self.current_position_size = sum(p['size'] for p in self.current_positions)
                    self.current_position_side = self.current_positions[0]['side']  # Assume all same side
                else:
                    self.current_position_side = 0
                    self.current_position_size = 0.0
            return response
        return None

    def place_market(self, side: str, sz: str) -> Optional[Any]:
        """Place a market order for the given side and size."""
        logger.info(f"Placing market order ({'DEMO' if self.is_demo else 'LIVE'}): {side} {sz} {self.symbol}")
        if self.is_demo:
            response = self._place_market_demo(side, sz)
        else:
            response = self._place_market_live(side, sz)
        self._log_order_placement(side, sz, response, is_live=not self.is_demo)
        return response

    def _place_market_demo(self, side: str, sz: str) -> Optional[Any]:
        """Place market order in demo mode using OKX REST API."""
        if not self.trade:
            logger.warning(f"DRY RUN: Would place {side} {sz} {self.symbol}")
            self._append_trade_record({'ts': datetime.utcnow(), 'event': 'order_placed', 'side': side, 'size': sz, 'price': None, 'pnl': None, 'fees': None})
            return None

        body = self._build_order_body(side, sz, "market")
        response = self.trade.place_order(body)
        logger.info(f"Order placement response: {response}")
        return response

    def _place_market_live(self, side: str, sz: str) -> Optional[Any]:
        """Place market order in live mode using CCXT."""
        try:
            params = {"tdMode": self.tdMode, "posSide": "long" if side == "buy" else "short"}
            response = self.exchange.create_order(self.symbol, "market", side, float(sz), None, params)
            logger.info(f"Order placement response: {response}")
            return response
        except Exception as e:
            logger.error(f"CCXT order placement failed: {e}")
            return None

    def _build_order_body(self, side: str, sz: str, ord_type: str) -> dict:
        """Build the order body for OKX API."""
        body = {
            "instId": self.symbol,
            "tdMode": self.tdMode,
            "side": side,
            "ordType": ord_type,
            "sz": str(sz),
        }
        pos_mode = self.cfg.get("posMode", "net_mode")
        if pos_mode == "long_short_mode":
            body["posSide"] = 'long' if side == 'buy' else 'short'
        return body

    def _log_order_placement(self, side: str, sz: str, response, is_live: bool = False) -> None:
        """Log order placement and append to trade history."""
        try:
            if is_live:
                px = response.get('price') if response else None
            else:
                px = None
                if response and response.get('data') and isinstance(response['data'], list) and len(response['data']) > 0:
                    px = response['data'][0].get('fillPx')

            # Calculate fees
            contract_size = self.cfg.get("contract_size", 1)
            amount = float(sz) * contract_size * (float(px) if px else 1.0)
            fees_paid_calc = self.fees * amount

            self._append_trade_record({
                'ts': datetime.utcnow(),
                'event': 'order_placed',
                'side': side,
                'size': sz,
                'price': px,
                'pnl': None,
                'fees': fees_paid_calc
            })
        except Exception as e:
            logger.debug(f"Could not append trade record: {e}")

    def get_balance(self, ccy: str = 'USDT') -> Optional[Any]:
        if self.is_demo:
            # Demo mode: use OKX REST API
            if not self.account:
                logger.warning("Account API client not available for demo mode.")
                return None
            response = self.account.get_account_balance(ccy=ccy)
            return response
        else:
            # Live mode: use CCXT
            try:
                balance = self.exchange.fetch_balance()
                return {
                    'code': '0',
                    'data': [{
                        'details': [{
                            'availBal': str(balance.get(ccy, {}).get('free', 0))
                        }]
                    }]
                }
            except Exception as e:
                logger.error(f"CCXT balance fetch failed: {e}")
                return None

    def cancel_all_orders(self, instId: Optional[str] = None) -> Optional[Dict[str, int]]:
        """Cancel all pending orders for the symbol or all if instId is None."""
        if self.is_demo:
            if not self.trade:
                logger.warning("Trade API client not available for demo mode.")
                return None
            if instId is None:
                logger.info("Cancelling all pending orders for all instruments")
                orders_response = self.trade.get_orders(instId=None)
            else:
                logger.info(f"Cancelling all pending orders for {instId}")
                orders_response = self.trade.get_orders(instId=instId)
            if orders_response and orders_response.get('code') == '0' and orders_response.get('data'):
                cancelled_count = 0
                for order in orders_response['data']:
                    ordId = order.get('ordId')
                    inst_id = order.get('instId', instId or self.symbol)
                    if ordId:
                        cancel_response = self.trade.cancel_order(inst_id, ordId=ordId)
                        if cancel_response and cancel_response.get('code') == '0':
                            cancelled_count += 1
                            logger.info(f"Cancelled order {ordId} for {inst_id}")
                        else:
                            logger.warning(f"Failed to cancel order {ordId} for {inst_id}")
                logger.info(f"Cancelled {cancelled_count} pending orders")
                return {'cancelled': cancelled_count}
            else:
                logger.info("No pending orders to cancel")
                return {'cancelled': 0}
        else:
            # Live mode: use CCXT
            try:
                if instId is None:
                    logger.info("Cancelling all pending orders for all instruments via CCXT")
                    orders = self.exchange.fetch_open_orders()
                else:
                    logger.info(f"Cancelling all pending orders for {instId} via CCXT")
                    orders = self.exchange.fetch_open_orders(instId)
                cancelled_count = 0
                for order in orders:
                    try:
                        self.exchange.cancel_order(order['id'], order['symbol'])
                        cancelled_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to cancel order {order['id']}: {e}")
                logger.info(f"Cancelled {cancelled_count} pending orders")
                return {'cancelled': cancelled_count}
            except Exception as e:
                logger.error(f"CCXT cancel all orders failed: {e}")
                return None

    def force_close_position(self, pos_side: str) -> Optional[Any]:
        """Force close an existing position."""
        if self.is_demo:
            if not self.trade:
                logger.warning("Trade API client not available for demo mode.")
                return None
            logger.info(f"Force closing {pos_side} position")
            response = self.trade.close_position(self.symbol, self.tdMode, pos_side)
            if response and response.get('code') == '0':
                # Log the exit
                self._append_trade_record({'ts': datetime.utcnow(), 'event': 'position_closed', 'side': pos_side, 'size': None, 'price': None, 'pnl': None, 'fees': None})
            return response
        else:
            # Live mode: use CCXT
            try:
                logger.info(f"Force closing {pos_side} position via CCXT")
                # CCXT close_position method
                response = self.exchange.close_position(self.symbol, {"posSide": pos_side})
                if response:
                    # Log the exit
                    self._append_trade_record({'ts': datetime.utcnow(), 'event': 'position_closed', 'side': pos_side, 'size': None, 'price': None, 'pnl': None, 'fees': None})
                return response
            except Exception as e:
                logger.error(f"CCXT force close failed: {e}")
                return None

    def place_trailing_stop(self, side: str, sz: str) -> Optional[Any]:
        """Place a trailing stop order."""
        logger.info(f"Placing trailing stop order ({'DEMO' if self.is_demo else 'LIVE'}): {side} {sz} {self.symbol}")
        if self.is_demo:
            return self._place_trailing_stop_demo(side, sz)
        else:
            return self._place_trailing_stop_live(side, sz)

    def _place_trailing_stop_demo(self, side: str, sz: str) -> Optional[Any]:
        """Place trailing stop order in demo mode using OKX REST API."""
        if not self.trade:
            logger.warning(f"DRY RUN: Would place trailing stop {side} {sz} {self.symbol}")
            return None

        body = {
            "instId": self.symbol,
            "tdMode": self.tdMode,
            "side": side,
            "ordType": "trailingStop",
            "sz": str(sz),
            "callbackRatio": f"{self.params.initial_trail_pct}",
        }
        try:
            response = self.trade.place_algo_order(body)
            logger.info(f"Trailing stop placement response: {response}")
            return response
        except Exception as e:
            logger.error(f"Trailing stop placement failed: {e}")
            return None

    def _place_trailing_stop_live(self, side: str, sz: str) -> Optional[Any]:
        """Place trailing stop order in live mode using CCXT."""
        try:
            # CCXT for OKX supports algo orders via create_order with ordType
            params = {
                "ordType": "trailingStop",
                "sz": str(sz),
                "callbackRatio": f"{self.params.initial_trail_pct}",
                "tdMode": self.tdMode,
                "posSide": "long" if side == "buy" else "short"
            }
            response = self.exchange.create_order(self.symbol, "market", side, float(sz), None, params)
            logger.info(f"Trailing stop placement response: {response}")
            return response
        except Exception as e:
            logger.error(f"CCXT trailing stop placement failed: {e}")
            return None

    def _append_trade_record(self, record: dict) -> None:
        """Append a trade event to the configured trade_history CSV.

        record keys expected: ts (datetime), event (str), side (str), size, price, pnl
        """
        path = self.cfg.get('paths', {}).get('trade_history')
        if not path:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        with _trade_history_lock: # Use the lock here
            write_header = not p.exists()
            with p.open('a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['ts', 'event', 'side', 'size', 'price', 'pnl', 'fees'])
                ts = record.get('ts')
                if hasattr(ts, 'isoformat'):
                    ts = ts.isoformat()
                writer.writerow([ts, record.get('event'), record.get('side'), record.get('size'), record.get('price'), record.get('pnl'), record.get('fees')])
