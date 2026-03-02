import ccxt
import pandas as pd
from datetime import datetime

def fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=1000, pages=1) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    all_ohlcv = []
    since = None

    for page in range(pages):
        print(f"📡 Hole Seite {page+1}/{pages} — {limit} Kerzen {symbol} {timeframe}...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_ohlcv = ohlcv + all_ohlcv
        since = ohlcv[0][0] - (limit * _ms_per_candle(timeframe))

    df = pd.DataFrame(all_ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("time").drop("timestamp", axis=1)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    print(f"✅ {len(df)} Kerzen total — {df.index[0]} bis {df.index[-1]}")
    return df

def _ms_per_candle(timeframe):
    mapping = {"1m":60000,"5m":300000,"15m":900000,"1h":3600000,"4h":14400000,"1d":86400000}
    return mapping.get(timeframe, 3600000)

def fetch_multi_timeframe(symbol="BTC/USDT") -> dict:
    return {
        "1h": fetch_ohlcv(symbol, "1h", limit=1000, pages=3),
        "4h": fetch_ohlcv(symbol, "4h", limit=1000, pages=1),
    }

if __name__ == "__main__":
    data = fetch_multi_timeframe()
    print("\n1H Kerzen:", len(data["1h"]))
    print("4H Kerzen:", len(data["4h"]))
