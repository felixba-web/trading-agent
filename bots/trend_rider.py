import sys
sys.path.insert(0, '/docker/trading-agent')
import pandas as pd
import numpy as np
from bots.base_bot import BaseBot

class TrendRider(BaseBot):
    EMA_FAST      = 21
    EMA_SLOW      = 55
    RSI_PERIOD    = 14
    RSI_MIN       = 45
    RSI_MAX       = 70
    ATR_PERIOD    = 14
    VOLUME_PERIOD = 20
    ST_MULTIPLIER = 3.0

    def __init__(self, capital):
        super().__init__("TrendRider", "bullish", capital)

    def compute_indicators(self, df):
        c, v = df["close"], df["volume"]
        d = df.copy()
        d["ema_fast"]  = c.ewm(span=self.EMA_FAST, adjust=False).mean()
        d["ema_slow"]  = c.ewm(span=self.EMA_SLOW, adjust=False).mean()
        d["rsi"]       = self._rsi(c)
        d["volume_ma"] = v.rolling(self.VOLUME_PERIOD).mean()
        d["volume_ok"] = v > d["volume_ma"]
        d["atr"]       = self._atr(d)
        d["st_up"], d["st_down"], d["supertrend"] = self._supertrend(d)
        return d

    def score_signal(self, row, prev, regime):
        score, details = 0, {}
        regime_ok = regime == "bullish"
        details["regime"] = 3 if regime_ok else 0
        score += details["regime"]
        ema_cross = (float(prev["ema_fast"]) <= float(prev["ema_slow"]) and
                     float(row["ema_fast"])  >  float(row["ema_slow"]))
        details["ema_cross"] = 3 if ema_cross else 0
        score += details["ema_cross"]
        st_ok = float(row["supertrend"]) == 1
        details["supertrend"] = 2 if st_ok else 0
        score += details["supertrend"]
        rsi = float(row["rsi"])
        details["rsi"] = 2 if self.RSI_MIN <= rsi <= self.RSI_MAX else 0
        score += details["rsi"]
        details["volume"] = 1 if bool(row["volume_ok"]) else 0
        score += details["volume"]
        details["ema_rising"] = 1 if float(row["ema_fast"]) > float(prev["ema_fast"]) else 0
        score += details["ema_rising"]
        action = "buy" if (regime_ok and ema_cross and score >= self.MIN_SCORE) else "hold"
        return {"score": score, "action": action,
                "reason": f"TrendRider {score}/13", "details": details, "rsi": rsi}

    def calculate_stops(self, row, action):
        atr = float(row["atr"])
        price = float(row["close"])
        return price - atr * self.ATR_STOP, price + atr * self.ATR_TARGET

    def _rsi(self, c):
        d = c.diff()
        g = d.clip(lower=0).ewm(com=self.RSI_PERIOD-1, adjust=False).mean()
        l = (-d).clip(lower=0).ewm(com=self.RSI_PERIOD-1, adjust=False).mean()
        return (100 - 100/(1 + g/l.replace(0, np.nan))).fillna(50)

    def _atr(self, df):
        h, l, c = df["high"], df["low"], df["close"]
        pc = c.shift(1)
        tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
        return tr.ewm(com=self.ATR_PERIOD-1, adjust=False).mean()

    def _supertrend(self, df):
        atr   = df["atr"]
        hl2   = (df["high"] + df["low"]) / 2
        upper = hl2 + self.ST_MULTIPLIER * atr
        lower = hl2 - self.ST_MULTIPLIER * atr
        close = df["close"]
        st_up   = pd.Series(index=df.index, dtype=float)
        st_down = pd.Series(index=df.index, dtype=float)
        trend   = pd.Series(index=df.index, dtype=float)
        for i in range(1, len(df)):
            st_up.iloc[i]   = max(lower.iloc[i], st_up.iloc[i-1]) if close.iloc[i-1] > st_up.iloc[i-1] else lower.iloc[i]
            st_down.iloc[i] = min(upper.iloc[i], st_down.iloc[i-1]) if close.iloc[i-1] < st_down.iloc[i-1] else upper.iloc[i]
            if close.iloc[i] > st_down.iloc[i-1]: trend.iloc[i] = 1
            elif close.iloc[i] < st_up.iloc[i-1]: trend.iloc[i] = -1
            else: trend.iloc[i] = trend.iloc[i-1]
        return st_up, st_down, trend
