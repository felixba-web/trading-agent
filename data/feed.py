"""
feed.py - Echte OHLCV Daten via CCXT
Binance Public API - kein Account nötig
"""

import ccxt
import pandas as pd
from datetime import datetime


def fetch_ohlcv(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
    exchange = ccxt.binance({
        "enableRateLimit": True,
    })

    print(f"📡 Hole {limit} Kerzen {symbol} {timeframe} von Binance...")

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("time")
    df = df.drop("timestamp", axis=1)

    print(f"✅ {len(df)} Kerzen geladen — {df.index[0]} bis {df.index[-1]}")
    return df


def fetch_multi_timeframe(symbol: str = "BTC/USDT") -> dict:
    """Holt 1H und 4H Daten gleichzeitig"""
    return {
        "1h": fetch_ohlcv(symbol, "1h", 500),
        "4h": fetch_ohlcv(symbol, "4h", 200),
    }


if __name__ == "__main__":
    data = fetch_multi_timeframe()
    print("\n📊 1H Letzte 3 Kerzen:")
    print(data["1h"].tail(3).to_string())
    print("\n📊 4H Letzte 3 Kerzen:")
    print(data["4h"].tail(3).to_string())
