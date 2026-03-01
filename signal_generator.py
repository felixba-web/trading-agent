"""
signal_generator.py - 3-Schichten System
Schicht 1: Regime (4H Ichimoku Kumo + EMA200)
Schicht 2: Signal (EMA21/55 + MACD + RSI + Volume)
Schicht 3: Risiko (ATR Stop/Target)
"""

import pandas as pd
import numpy as np


class SignalGenerator:

    # Schicht 1 - Regime
    ICHIMOKU_CONVERSION = 9
    ICHIMOKU_BASE       = 26
    ICHIMOKU_SPAN_B     = 52
    EMA_MACRO           = 200

    # Schicht 2 - Signal
    EMA_FAST     = 21
    EMA_SLOW     = 55
    MACD_FAST    = 12
    MACD_SLOW    = 26
    MACD_SIGNAL  = 9
    RSI_PERIOD   = 14
    RSI_BUY_MIN  = 40
    RSI_BUY_MAX  = 65
    RSI_SELL_MIN = 35
    RSI_SELL_MAX = 60
    VOLUME_PERIOD = 20

    # Schicht 3 - Risiko
    ATR_PERIOD = 14
    ATR_STOP   = 1.5
    ATR_TARGET = 2.5

    def compute_regime(self, df_4h: pd.DataFrame) -> pd.DataFrame:
        high  = df_4h["high"]
        low   = df_4h["low"]
        close = df_4h["close"]

        conv   = (high.rolling(self.ICHIMOKU_CONVERSION).max() +
                  low.rolling(self.ICHIMOKU_CONVERSION).min()) / 2
        base   = (high.rolling(self.ICHIMOKU_BASE).max() +
                  low.rolling(self.ICHIMOKU_BASE).min()) / 2
        span_a = ((conv + base) / 2).shift(self.ICHIMOKU_BASE)
        span_b = ((high.rolling(self.ICHIMOKU_SPAN_B).max() +
                   low.rolling(self.ICHIMOKU_SPAN_B).min()) / 2).shift(self.ICHIMOKU_BASE)

        df = df_4h.copy()
        df["kumo_top"]    = pd.concat([span_a, span_b], axis=1).max(axis=1)
        df["kumo_bottom"] = pd.concat([span_a, span_b], axis=1).min(axis=1)
        df["ema200"]      = close.ewm(span=self.EMA_MACRO, adjust=False).mean()

        def get_regime(row):
            if pd.isna(row["kumo_top"]) or pd.isna(row["kumo_bottom"]):
                return "neutral"
            if row["close"] > row["kumo_top"]:
                return "bullish"
            elif row["close"] < row["kumo_bottom"]:
                return "bearish"
            else:
                return "sideways"

        df["regime"] = df.apply(get_regime, axis=1)
        return df

    def compute_signals(self, df_1h: pd.DataFrame) -> pd.DataFrame:
        close  = df_1h["close"]
        volume = df_1h["volume"]
        df     = df_1h.copy()

        df["ema_fast"] = close.ewm(span=self.EMA_FAST, adjust=False).mean()
        df["ema_slow"] = close.ewm(span=self.EMA_SLOW, adjust=False).mean()

        macd_fast        = close.ewm(span=self.MACD_FAST, adjust=False).mean()
        macd_slow        = close.ewm(span=self.MACD_SLOW, adjust=False).mean()
        df["macd"]       = macd_fast - macd_slow
        df["macd_signal"]= df["macd"].ewm(span=self.MACD_SIGNAL, adjust=False).mean()

        df["rsi"]       = self._rsi(close, self.RSI_PERIOD)
        df["volume_ma"] = volume.rolling(self.VOLUME_PERIOD).mean()
        df["volume_ok"] = volume > df["volume_ma"]
        df["atr"]       = self._atr(df["high"], df["low"], close, self.ATR_PERIOD)

        return df

    def get_signal(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame, idx: int) -> dict:
        row   = df_1h.iloc[idx]
        prev  = df_1h.iloc[idx - 1]
        price = float(row["close"])
        ts    = df_1h.index[idx]

        regime = self._get_regime_at(df_4h, ts)

        if regime == "sideways":
            return self._make_signal("hold", price, row, regime, "Kumo Seitwarts")

        ema_up   = float(prev["ema_fast"]) <= float(prev["ema_slow"]) and \
                   float(row["ema_fast"])  >  float(row["ema_slow"])
        ema_down = float(prev["ema_fast"]) >= float(prev["ema_slow"]) and \
                   float(row["ema_fast"])  <  float(row["ema_slow"])

        macd_up   = float(prev["macd"]) <= float(prev["macd_signal"]) and \
                    float(row["macd"])  >  float(row["macd_signal"])
        macd_down = float(prev["macd"]) >= float(prev["macd_signal"]) and \
                    float(row["macd"])  <  float(row["macd_signal"])

        rsi       = float(row["rsi"])
        volume_ok = bool(row["volume_ok"])
        atr       = float(row["atr"])

        if (ema_up and macd_up and volume_ok and
                regime == "bullish" and
                self.RSI_BUY_MIN <= rsi <= self.RSI_BUY_MAX):
            return self._make_signal("buy", price, row, regime, "3/3 Schichten",
                                     stop=price - atr * self.ATR_STOP,
                                     target=price + atr * self.ATR_TARGET)

        if (ema_down and macd_down and volume_ok and
                regime == "bearish" and
                self.RSI_SELL_MIN <= rsi <= self.RSI_SELL_MAX):
            return self._make_signal("sell", price, row, regime, "3/3 Schichten",
                                     stop=price + atr * self.ATR_STOP,
                                     target=price - atr * self.ATR_TARGET)

        return self._make_signal("hold", price, row, regime, "Keine Bestatigung")

    def _get_regime_at(self, df_4h, ts):
        past = df_4h[df_4h.index <= ts]
        if past.empty or "regime" not in past.columns:
            return "neutral"
        return str(past.iloc[-1]["regime"])

    def _make_signal(self, action, price, row, regime, reason,
                     stop=None, target=None):
        return {
            "action":    action,
            "price":     price,
            "regime":    regime,
            "reason":    reason,
            "rsi":       float(row.get("rsi", 0)),
            "macd":      float(row.get("macd", 0)),
            "atr":       float(row.get("atr", 0)),
            "ema_fast":  float(row.get("ema_fast", 0)),
            "ema_slow":  float(row.get("ema_slow", 0)),
            "volume_ok": bool(row.get("volume_ok", False)),
            "stop":      stop,
            "target":    target,
        }

    @staticmethod
    def _rsi(close, period):
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50)

    @staticmethod
    def _atr(high, low, close, period):
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, adjust=False).mean()
