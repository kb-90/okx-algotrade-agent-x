import csv
import json
from pathlib import Path
from datetime import datetime
import threading


_lock = threading.Lock()


def _get_paths(cfg):
    paths = cfg.get('paths', {}) if cfg else {}
    trade_path = paths.get('trade_history', 'state/trade_history.csv')
    summary_path = str(Path(trade_path).parent / 'trade_summary.json')
    return Path(trade_path), Path(summary_path)


def append_event(cfg, record: dict):
    """Append an event record to trade_history CSV and update summary if pnl present."""
    trade_path, summary_path = _get_paths(cfg)
    trade_path.parent.mkdir(parents=True, exist_ok=True)
    header = ['ts', 'event', 'side', 'size', 'price', 'pnl', 'ordId', 'instId']
    with _lock:
        write_header = not trade_path.exists()
        with trade_path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            ts = record.get('ts') or datetime.utcnow()
            if hasattr(ts, 'isoformat'):
                ts = ts.isoformat()
            row = [ts, record.get('event'), record.get('side'), record.get('size'), record.get('price'), record.get('pnl'), record.get('ordId'), record.get('instId')]
            writer.writerow(row)

        # update summary if pnl present and numeric
        pnl = record.get('pnl')
        try:
            pnl_f = float(pnl) if pnl is not None else None
        except Exception:
            pnl_f = None
        if pnl_f is not None:
            _update_summary(summary_path, pnl_f, win=(pnl_f > 0))


def _update_summary(summary_path: Path, pnl: float, win: bool):
    summary = {"total_trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding='utf-8') or '{}')
        except Exception:
            summary = {"total_trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}

    summary['total_trades'] = summary.get('total_trades', 0) + 1
    if win:
        summary['wins'] = summary.get('wins', 0) + 1
    else:
        summary['losses'] = summary.get('losses', 0) + 1
    summary['total_pnl'] = float(summary.get('total_pnl', 0.0)) + float(pnl)

    try:
        summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    except Exception:
        pass


def read_summary(cfg):
    _, summary_path = _get_paths(cfg)
    if not summary_path.exists():
        return {"total_trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}
    try:
        return json.loads(summary_path.read_text(encoding='utf-8') or '{}')
    except Exception:
        return {"total_trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}
