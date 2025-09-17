import os
import json
import base64
import hmac
import hashlib
import logging
import threading
import time
import websocket
import requests
from datetime import datetime, timezone
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from ratelimit import limits, sleep_and_retry

# Use the logger from the agent's utilsa
from .utils import logger

# ####################################################################
# Websocket Client
# ####################################################################
# Replace your OKXWebSocketClient class with this version (keeps your file structure intact)
# Ensures correct Demo WS URLs (wspap + brokerId=9999) and avoids requiring creds for public-only subs.

class OKXWebSocketClient:
    def __init__(self, cfg, message_queue, public_subscriptions, private_subscriptions):
        self.demo_mode = cfg['runtime'].get('demo', True)
        self.message_queue = message_queue
        self.public_ws = None
        self.private_ws = None
        self.public_thread = None
        self.private_thread = None
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.public_subscriptions = public_subscriptions or []
        self.private_subscriptions = private_subscriptions or []
        self.ping_interval = 20
        self.successful_subscriptions_count = 0
        self.total_subscriptions = len(self.public_subscriptions) + len(self.private_subscriptions)
        self.retry_count = 0
        self.max_retries = 5

        # Load creds (needed only for private subs)
        self.api_key = os.getenv("YOUR_API_KEY")
        self.secret_key = os.getenv("YOUR_SECRET_KEY")
        self.passphrase = os.getenv("YOUR_PASSPHRASE")
        if self.demo_mode:
            logger.info("WS: Demo mode enabled.")
            self.private_url = cfg['runtime'].get('ws_private_demo')
            self.public_url = cfg['runtime'].get('ws_public_demo')
        else:
            logger.info("WS: Live mode enabled.")
            self.private_url = cfg['runtime'].get('ws_private')
            self.public_url = cfg['runtime'].get('ws_public')

        # Normalize to str for safety
        self.api_key = str(self.api_key) if self.api_key is not None else None
        self.secret_key = str(self.secret_key) if self.secret_key is not None else None
        self.passphrase = str(self.passphrase) if self.passphrase is not None else None

        # Build correct WS URLs (fix demo URLs and add brokerId=9999 automatically)
        self.public_url = self._ensure_ws_url(self.public_url, is_private=False)
        self.private_url = self._ensure_ws_url(self.private_url, is_private=True)

        # Only enforce creds if private subs requested
        if self.private_subscriptions and not all([self.api_key, self.secret_key, self.passphrase]):
            raise RuntimeError("WS: Private subscriptions requested but API credentials are missing.")

        if not self.private_subscriptions and not all([self.api_key, self.secret_key, self.passphrase]):
            logger.info("WS: No private subscriptions requested; proceeding without API credentials.")

    def _ensure_ws_url(self, url: str, is_private: bool) -> str:
        """
        Ensure correct WS endpoint per environment.
        - Demo WS must use wspap.okx.com:8443 with brokerId=9999 for private only.
        - Live WS should use ws.okx.com:8443.
        - For public candles, use /business endpoint.
        """
        if self.demo_mode:
            if is_private:
                default = f"wss://wspap.okx.com:8443/ws/v5/private?brokerId=9999"
            else:
                default = f"wss://wspap.okx.com:8443/ws/v5/business"
            if not url:
                return default
            # Prepend host if only a path provided
            if url.startswith("/ws/"):
                url = "wss://wspap.okx.com:8443" + url
            # Ensure demo host
            if "wspap.okx.com" not in url:
                # If someone passed a live URL by mistake, switch it
                url = url.replace("wss://ws.okx.com", "wss://wspap.okx.com:8443")
                if not url.startswith("wss://wspap.okx.com"):
                    # Fallback to default if unknown host
                    url = default
            # For public, ensure /business endpoint
            if not is_private and "/ws/v5/public" in url:
                url = url.replace("/ws/v5/public", "/ws/v5/business")
            # Ensure brokerId only for private
            if is_private and "brokerId=" not in url:
                sep = "&" if "?" in url else "?"
                url += f"{sep}brokerId=9999"
            return url
        else:
            if is_private:
                default = f"wss://ws.okx.com:8443/ws/v5/private"
            else:
                default = f"wss://ws.okx.com:8443/ws/v5/public"
            if not url:
                return default
            # Prepend host if only a path provided
            if url.startswith("/ws/"):
                url = "wss://ws.okx.com:8443" + url
            # For public, ensure /public endpoint
            if not is_private and "/ws/v5/business" in url:
                url = url.replace("/ws/v5/business", "/ws/v5/public")
            return url

    def _generate_signature(self, timestamp):
        message = timestamp + 'GET' + '/users/self/verify'
        if not self.secret_key:
            raise RuntimeError("WS: Missing secret_key for signature.")
        mac = hmac.new(
            bytes(self.secret_key, encoding='utf8'),
            bytes(message, encoding='utf-8'),
            digestmod=hashlib.sha256,
        )
        return base64.b64encode(mac.digest()).decode('utf-8')

    def _on_public_message(self, ws, message):
        if not message or not message.strip():
            return
        if message == 'ping':
            ws.send('pong')
            return
        elif message == 'pong':
            return
        try:
            data = json.loads(message)
            if data.get("event") == "subscribe":
                logger.info(f"Subscribed to {data.get('arg')} on Public WS.")
                self.successful_subscriptions_count += 1
                if self.successful_subscriptions_count >= self.total_subscriptions:
                    self.ready_event.set()
            elif data.get("event") == "error":
                logger.error(f"Public WS error: {data.get('msg')} (code: {data.get('code')})")
            self.message_queue.put(data)
        except Exception as e:
            logger.error(f"Public WS message error: {e}")

    def _on_private_message(self, ws, message):
        if not message or not message.strip():
            return
        if message == 'ping':
            ws.send('pong')
            return
        elif message == 'pong':
            return
        try:
            data = json.loads(message)
            if data.get("event") == "login" and data.get("code") == "0":
                logger.info("Private WS login successful.")
                if self.private_subscriptions:
                    self._subscribe_to_private_channels(ws)
            elif data.get("event") == "subscribe":
                logger.info(f"Subscribed to {data.get('arg')} on Private WS.")
                self.successful_subscriptions_count += 1
                if self.successful_subscriptions_count >= self.total_subscriptions:
                    self.ready_event.set()
            elif data.get("event") == "error":
                logger.error(f"Private WS error: {data.get('msg')} (code: {data.get('code')})")
            self.message_queue.put(data)
        except Exception as e:
            logger.error(f"Private WS message error: {e}")

    def _on_error(self, ws, error):
        conn_type = "Private" if ws == self.private_ws else "Public"
        logger.error(f"{conn_type} WS Error: {error}")
        self.ready_event.clear()

    def _on_close(self, ws, close_status_code, close_msg):
        conn_type = "Private" if ws == self.private_ws else "Public"
        logger.info(f"{conn_type} WS closed: {close_msg} (code: {close_status_code})")
        if not self.stop_event.is_set():
            self._reconnect()

    def _on_public_open(self, ws):
        logger.info(f"Public WS opened. URL: {self.public_url}")
        if self.public_subscriptions:
            self._subscribe_to_public_channels(ws)

    def _on_private_open(self, ws):
        logger.info(f"Private WS opened. URL: {self.private_url}")
        timestamp = str(time.time())
        sign = self._generate_signature(timestamp)
        login_payload = {"op": "login", "args": [{"apiKey": self.api_key, "passphrase": self.passphrase, "timestamp": timestamp, "sign": sign}]}
        ws.send(json.dumps(login_payload))

    def _subscribe_to_public_channels(self, ws):
        payload = {"op": "subscribe", "args": self.public_subscriptions}
        ws.send(json.dumps(payload))
        logger.info(f"Sent public subscription: {payload}")

    def _subscribe_to_private_channels(self, ws):
        payload = {"op": "subscribe", "args": self.private_subscriptions}
        ws.send(json.dumps(payload))
        logger.info(f"Sent private subscription: {payload}")

    def _send_ping(self, ws, conn_type):
        while not self.stop_event.is_set():
            try:
                if ws and ws.sock and ws.sock.connected:
                    ws.send("ping")
                time.sleep(self.ping_interval)
            except Exception as e:
                logger.error(f"{conn_type} WS ping error: {e}")
                break

    def _reconnect(self):
        self.ready_event.clear()
        self.successful_subscriptions_count = 0
        if self.retry_count < self.max_retries:
            self.retry_count += 1
            wait_time = min(2 ** self.retry_count, 60)
            logger.info(f"Reconnecting WS in {wait_time}s (Attempt {self.retry_count}/{self.max_retries})")
            time.sleep(wait_time)
            self.start_threads()
        else:
            logger.error("Max WS reconnect attempts reached. Stopping.")
            self.stop()

    def wait_for_ready(self, timeout=30):
        return self.ready_event.wait(timeout=timeout)

    def start(self):
        self.stop_event.clear()
        self.start_threads()

    def start_threads(self):
        if self.public_subscriptions and (not self.public_thread or not self.public_thread.is_alive()):
            self.public_thread = threading.Thread(target=self._run_public, daemon=True)
            self.public_thread.start()
        if self.private_subscriptions and (not self.private_thread or not self.private_thread.is_alive()):
            self.private_thread = threading.Thread(target=self._run_private, daemon=True)
            self.private_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.public_ws:
            self.public_ws.close()
        if self.private_ws:
            self.private_ws.close()
        if self.public_thread and self.public_thread.is_alive():
            self.public_thread.join(timeout=5)
        if self.private_thread and self.private_thread.is_alive():
            self.private_thread.join(timeout=5)

    def _run_public(self):
        while not self.stop_event.is_set():
            try:
                logger.info("Starting Public WS...")
                self.public_ws = websocket.WebSocketApp(
                    self.public_url, on_open=self._on_public_open, on_message=self._on_public_message,
                    on_error=self._on_error, on_close=self._on_close
                )
                ping_thread = threading.Thread(target=lambda: self._send_ping(self.public_ws, "Public"), daemon=True)
                ping_thread.start()
                self.public_ws.run_forever(ping_timeout=self.ping_interval + 5)
            except Exception as e:
                logger.error(f"Public WS run error: {e}")
            if not self.stop_event.is_set():
                time.sleep(10)

    def _run_private(self):
        while not self.stop_event.is_set():
            try:
                logger.info("Starting Private WS...")
                self.private_ws = websocket.WebSocketApp(
                    self.private_url, on_open=self._on_private_open, on_message=self._on_private_message,
                    on_error=self._on_error, on_close=self._on_close
                )
                ping_thread = threading.Thread(target=lambda: self._send_ping(self.private_ws, "Private"), daemon=True)
                ping_thread.start()
                self.private_ws.run_forever(ping_timeout=self.ping_interval + 5)
            except Exception as e:
                logger.error(f"Private WS run error: {e}")
            if not self.stop_event.is_set():
                time.sleep(10)

# ####################################################################
# REST API Clients
# ####################################################################
class OKXBaseAPI:
    def __init__(self, cfg):
        self.session = requests.Session()
        retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)

        self.demo_mode = cfg['runtime'].get('demo', True)
        self.api_key = os.getenv("YOUR_API_KEY")
        self.secret_key = os.getenv("YOUR_SECRET_KEY")
        self.passphrase = os.getenv("YOUR_PASSPHRASE")
        if self.demo_mode:
            self.base_url = cfg['runtime']['base_url_demo']
        else:
            self.base_url = cfg['runtime']['base_url']

        if not all([self.api_key, self.secret_key, self.passphrase]):
            mode = "demo" if self.demo_mode else "live"
            raise ValueError(f"API credentials for {mode} mode are missing. Please check your .env file.")

    def _get_headers(self, method, request_path, body_str):
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        message = timestamp + method.upper() + request_path + body_str
        mac = hmac.new(bytes(self.secret_key, encoding="utf8"), bytes(message, encoding="utf-8"), digestmod=hashlib.sha256)
        sign = base64.b64encode(mac.digest()).decode("utf-8")
        headers = {
            "Content-Type": "application/json", "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": sign, "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
        }
        if self.demo_mode:
            headers["x-simulated-trading"] = "1"
        return headers

    @sleep_and_retry
    @limits(calls=19, period=1)
    def _request(self, method, endpoint, params=None, body=None, api_name="API"):
        if not self.api_key: # Guard against requests if keys are missing
            logger.error(f"{api_name}: Cannot make authenticated request, API key is missing.")
            return None

        url = f"{self.base_url}{endpoint}"
        body_str = json.dumps(body) if body else ""
        request_path = endpoint
        if params:
            query_string = "?" + "&".join([f"{k}={v}" for k, v in params.items()])
            request_path += query_string

        logger.info(f"[{api_name}] Request: {method} {url} | Params: {params} | Body: {body_str}")
        headers = self._get_headers(method, request_path, body_str)
        logger.debug(f"Request: {method} {url} Headers: {headers} Body: {body_str}")
        try:
            response = self.session.request(method, url, headers=headers, params=params, data=body_str, timeout=30)
            logger.debug(f"Response status: {response.status_code} Response text: {response.text}")
            result = response.json()
            logger.info(f"[{api_name}] Response: {result}")
            response.raise_for_status()
            if result.get("code") != "0":
                logger.warning(f"{api_name} Error: {result.get('msg')} (Code: {result.get('code')}) for path {request_path}")
            return result
        except requests.exceptions.HTTPError as e:
            logger.error(f"{api_name} HTTP Error: {e.response.status_code} {e.response.text}")
        except Exception as e:
            logger.error(f"{api_name} request failed: {e}")
        return None

class OKXPublicAPI(OKXBaseAPI):
    @sleep_and_retry
    @limits(calls=19, period=1)
    def _request(self, method, endpoint, params=None, body=None, api_name="PublicAPI"):
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, params=params, json=body, timeout=30)
            response.raise_for_status()
            result = response.json()
            if result.get("code") != "0":
                logger.warning(f"{api_name} Error: {result.get('msg')} (Code: {result.get('code')}) for path {endpoint}")
            return result
        except requests.exceptions.HTTPError as e:
            logger.error(f"{api_name} HTTP Error: {e.response.status_code} {e.response.text}")
        except Exception as e:
            logger.error(f"{api_name} request failed: {e}")
        return None

class OKXMarketAPI(OKXPublicAPI):
    def get_historical_candles(self, symbol, timeframe, limit, after_ts=None, before_ts=None):
        params = {"instId": symbol, "bar": timeframe, "limit": str(limit)}
        if after_ts:
            params["after"] = after_ts
        if before_ts:
            params["before"] = before_ts
        # Use the public, unauthenticated endpoint for history
        return self._request("GET", "/api/v5/market/history-candles", params=params, api_name="MarketAPI")

    def get_instrument_details(self, symbol, instType="SWAP"):
        params = {"instType": instType, "instId": symbol}
        return self._request("GET", "/api/v5/public/instruments", params=params, api_name="MarketAPI")

class OKXTradeAPI(OKXBaseAPI):
    def set_position_mode(self, mode):
        return self._request("POST", "/api/v5/account/set-position-mode", body={"posMode": mode}, api_name="TradeAPI")

    def set_leverage(self, symbol, leverage, mgn_mode, posSide=None):
        body = {"instId": symbol, "lever": str(leverage), "mgnMode": mgn_mode}
        if posSide: body["posSide"] = posSide
        return self._request("POST", "/api/v5/account/set-leverage", body=body, api_name="TradeAPI")

    def place_order(self, body):
        return self._request("POST", "/api/v5/trade/order", body=body, api_name="TradeAPI")

    def place_algo_order(self, body):
        return self._request("POST", "/api/v5/trade/order-algo", body=body, api_name="TradeAPI")

    def get_account_balance(self, ccy='USDT'):
        return self._request("GET", "/api/v5/account/balance", params={"ccy": ccy}, api_name="TradeAPI")

    def close_position(self, symbol, mgn_mode, posSide):
        body = {"instId": symbol, "mgnMode": mgn_mode, "posSide": posSide}
        return self._request("POST", "/api/v5/trade/close-position", body=body, api_name="TradeAPI")

    def get_positions(self, instId=None, instType="SWAP"):
        params = {"instType": instType}
        if instId:
            params["instId"] = instId
        return self._request("GET", "/api/v5/account/positions", params=params, api_name="TradeAPI")

    def get_orders(self, instId=None, instType="SWAP", state="live"):
        params = {"instType": instType, "state": state}
        if instId is not None:
            params["instId"] = instId
        return self._request("GET", "/api/v5/trade/orders-pending", params=params, api_name="TradeAPI")

    def cancel_order(self, instId, ordId=None, clOrdId=None):
        body = {"instId": instId}
        if ordId:
            body["ordId"] = ordId
        if clOrdId:
            body["clOrdId"] = clOrdId
        return self._request("POST", "/api/v5/trade/cancel-order", body=body, api_name="TradeAPI")

# ####################################################################
# Main Client Wrapper (for compatibility with the rest of the bot)
# ####################################################################
class OKXAPIClient:
    def __init__(self, cfg):
        self.trade = OKXTradeAPI(cfg)
        self.market = OKXMarketAPI(cfg)
        # Methods from AccountAPI are in OKXTradeAPI in this implementation
        self.account = self.trade 
        self.public_data = OKXPublicAPI(cfg)