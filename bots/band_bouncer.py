import sys
sys.path.insert(0, '/docker/trading-agent')
import pandas as pd
import numpy as np
from bots.base_bot import BaseBot

class BandBouncer(BaseBot):
    BB_PERIOD     = 20
    BB_STD        = 2.0
    RSI_PERIOD    = 14
    RSI_OB        = 65
    RSI_OS        = 35
    ATR_PERIOD    = 14
    VOLUME_PERIOD = 20

    def __init__(self, capital):
        super().__init__("BandBouncer", "sideways", capital)

    def compute_indicators(self, df):
        c, v = df["close"], df["volume"]
        d = df.copy()
        d["bb_mid"]   = c.rolling(self.BB_PERIOD).mean()
        bb_std        = c.rolling(self.BB_PERIOD).std()
        d["bb_upper"] = d["bb_mid"] + self.BB_STD * bb_std
        d["bb_lower"] = d["bb_mid"] - self.BB_STD * bb_std
        d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / d["bb_mid"]
        d["bb_flat"]  = d["bb_width"] < d["bb_width"].rolling(50).mean()
        d["rsi"]      = self._rsi(c)
        d["volume_ma"]= v.rolling(self.VOLUME_PERIOD).mean()
        d["volume_ok"]= v > d["volume_ma"]
        d["atr"]      = self._atr(d)
        return d

    def score_signal(self, row, prev, regime):
        score, details = 0, {}
        regime_ok = regime == "sideways"
        details["regime"] = 3 if regime_ok else 0
        score += details["regime"]
        price    = float(row["close"])
        bb_lower = float(row["bb_lower"])
        bb_upper = float(row["bb_upper"])
        rsi      = float(row["rsi"])
        near_lower = price <= bb_lower * 1.005
        near_upper = price >= bb_upper * 0.995
        reversal_up   = float(row["close"]) > float(prev["close"])
        reversal_down = float(row["close"]) < float(prev["close"])
        if near_lower and reversal_up:
            action = "buy"
            details["bb_signal"] = 3
        elif near_upper and reversal_down:
            action = "sell"
            details["bb_signal"] = 3
        else:
            action = "hold"
            details["bb_signal"] = 0
        score += details["bb_signal"]
        bb_flat = bool(row["bb_flat"]) if not pd.isna(row["bb_flat"]) else False
        details["bb_flat"] = 2 if bb_flat else 0
        score += details["bb_flat"]
        rsi_ok = (action == "buy"  and rsi < self.RSI_OS + 15) or \
                 (action == "sell" and rsi > self.RSI_OB - 15)
        details["rsi"] = 2 if rsi_ok else 0
        score += details["rsi"]
        details["volume"] = 1 if bool(row["volume_ok"]) else 0
        score += details["volume"]
        rsi_turning = (action == "buy"  and float(row["rsi"]) > float(prev["rsi"])) or \
                      (action == "sell" and float(row["rsi"]) < float(prev["rsi"]))
        details["rsi_turning"] = 1 if rsi_turning else 0
        score += details["rsi_turning"]
        if not (regime_ok and action != "hold" and score >= self.MIN_SCORE):
            action = "hold"
        return {"score": score, "action": action,
                "reason": f"BandBouncer {score}/13", "details": details, "rsi": rsi}

    def calculate_stops(self, row, action):
        atr    = float(row["atr"])
        price  = float(row["close"])
        bb_mid = float(row["bb_mid"])
        if action == "buy":
            stop   = price - atr * self.ATR_STOP
            target = max(bb_mid, price + atr * 1.5)
        else:
            stop   = price + atr * self.ATR_STOP
            target = min(bb_mid, price - atr * 1.5)
        return stop, target

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
