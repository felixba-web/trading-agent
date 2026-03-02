import sys
sys.path.insert(0, '/docker/trading-agent')
import pandas as pd
import numpy as np
from bots.base_bot import BaseBot

class DivHunter(BaseBot):
    RSI_PERIOD    = 14
    MACD_FAST     = 12
    MACD_SLOW     = 26
    MACD_SIGNAL   = 9
    ATR_PERIOD    = 14
    VOLUME_PERIOD = 20
    DIV_LOOKBACK  = 5
    MIN_SCORE     = 10

    def __init__(self, capital):
        super().__init__("DivHunter", "bearish", capital)

    def compute_indicators(self, df):
        c, v = df["close"], df["volume"]
        d = df.copy()
        d["rsi"]         = self._rsi(c)
        mf               = c.ewm(span=self.MACD_FAST, adjust=False).mean()
        ms               = c.ewm(span=self.MACD_SLOW, adjust=False).mean()
        d["macd"]        = mf - ms
        d["macd_signal"] = d["macd"].ewm(span=self.MACD_SIGNAL, adjust=False).mean()
        d["macd_hist"]   = d["macd"] - d["macd_signal"]
        d["volume_ma"]   = v.rolling(self.VOLUME_PERIOD).mean()
        d["volume_ok"]   = v > d["volume_ma"]
        d["atr"]         = self._atr(d)
        d["bearish_div"] = self._bearish_divergence(d)
        d["bullish_div"] = self._bullish_divergence(d)
        return d

    def _bearish_divergence(self, df):
        result = pd.Series(False, index=df.index)
        n = self.DIV_LOOKBACK
        for i in range(n, len(df)):
            wp = df["close"].iloc[i-n:i+1]
            wr = df["rsi"].iloc[i-n:i+1]
            if (df["close"].iloc[i] > wp.iloc[:-1].max() and
                    df["rsi"].iloc[i] < wr.iloc[:-1].max()):
                result.iloc[i] = True
        return result

    def _bullish_divergence(self, df):
        result = pd.Series(False, index=df.index)
        n = self.DIV_LOOKBACK
        for i in range(n, len(df)):
            wp = df["close"].iloc[i-n:i+1]
            wr = df["rsi"].iloc[i-n:i+1]
            if (df["close"].iloc[i] < wp.iloc[:-1].min() and
                    df["rsi"].iloc[i] > wr.iloc[:-1].min()):
                result.iloc[i] = True
        return result

    def score_signal(self, row, prev, regime):
        score, details = 0, {}
        regime_ok = regime == "bearish"
        details["regime"] = 3 if regime_ok else 0
        score += details["regime"]
        bearish_div = bool(row["bearish_div"])
        if regime == "bearish":
            action = "sell" if bearish_div else "hold"
        else:
            bullish_div = bool(row["bullish_div"])
            if bearish_div: action = "sell"
            elif bullish_div: action = "buy"
            else: action = "hold"
        details["divergence"] = 3 if action != "hold" else 0
        score += details["divergence"]
        macd_ok = (action == "sell" and float(row["macd_hist"]) < 0) or \
                  (action == "buy"  and float(row["macd_hist"]) > 0)
        details["macd"] = 2 if macd_ok else 0
        score += details["macd"]
        rsi = float(row["rsi"])
        rsi_ok = (action == "sell" and rsi > 55) or (action == "buy" and rsi < 45)
        details["rsi"] = 2 if rsi_ok else 0
        score += details["rsi"]
        details["volume"] = 1 if bool(row["volume_ok"]) else 0
        score += details["volume"]
        macd_cross = (action == "sell" and
                      float(prev["macd"]) >= float(prev["macd_signal"]) and
                      float(row["macd"])  <  float(row["macd_signal"])) or \
                     (action == "buy" and
                      float(prev["macd"]) <= float(prev["macd_signal"]) and
                      float(row["macd"])  >  float(row["macd_signal"]))
        details["macd_cross"] = 1 if macd_cross else 0
        score += details["macd_cross"]
        if not (regime_ok and action != "hold" and score >= self.MIN_SCORE):
            action = "hold"
        return {"score": score, "action": action,
                "reason": f"DivHunter {score}/13", "details": details, "rsi": rsi}

    def calculate_stops(self, row, action):
        atr   = float(row["atr"])
        price = float(row["close"])
        if action == "sell":
            return price + atr * self.ATR_STOP, price - atr * self.ATR_TARGET
        else:
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
