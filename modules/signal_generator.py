import pandas as pd
import numpy as np

class SignalGenerator:
    ICHIMOKU_CONVERSION = 9
    ICHIMOKU_BASE       = 26
    ICHIMOKU_SPAN_B     = 52
    EMA_MACRO           = 200
    EMA_FAST            = 21
    EMA_SLOW            = 55
    MACD_FAST           = 12
    MACD_SLOW           = 26
    MACD_SIGNAL_P       = 9
    RSI_PERIOD          = 14
    RSI_BUY_MIN         = 40
    RSI_BUY_MAX         = 65
    RSI_SELL_MIN        = 35
    RSI_SELL_MAX        = 60
    VOLUME_PERIOD       = 20
    ATR_PERIOD          = 14
    ATR_STOP            = 1.5
    ATR_TARGET          = 2.5

    def compute_regime(self, df):
        h, l, c = df["high"], df["low"], df["close"]
        conv   = (h.rolling(self.ICHIMOKU_CONVERSION).max() + l.rolling(self.ICHIMOKU_CONVERSION).min()) / 2
        base   = (h.rolling(self.ICHIMOKU_BASE).max() + l.rolling(self.ICHIMOKU_BASE).min()) / 2
        span_a = ((conv + base) / 2).shift(self.ICHIMOKU_BASE)
        span_b = ((h.rolling(self.ICHIMOKU_SPAN_B).max() + l.rolling(self.ICHIMOKU_SPAN_B).min()) / 2).shift(self.ICHIMOKU_BASE)
        d = df.copy()
        d["kumo_top"]    = pd.concat([span_a, span_b], axis=1).max(axis=1)
        d["kumo_bottom"] = pd.concat([span_a, span_b], axis=1).min(axis=1)
        d["ema200"]      = c.ewm(span=self.EMA_MACRO, adjust=False).mean()
        def regime(row):
            if pd.isna(row["kumo_top"]): return "neutral"
            if row["close"] > row["kumo_top"]: return "bullish"
            if row["close"] < row["kumo_bottom"]: return "bearish"
            return "sideways"
        d["regime"] = d.apply(regime, axis=1)
        return d

    def compute_signals(self, df):
        c, v = df["close"], df["volume"]
        d = df.copy()
        d["ema_fast"]    = c.ewm(span=self.EMA_FAST, adjust=False).mean()
        d["ema_slow"]    = c.ewm(span=self.EMA_SLOW, adjust=False).mean()
        mf               = c.ewm(span=self.MACD_FAST, adjust=False).mean()
        ms               = c.ewm(span=self.MACD_SLOW, adjust=False).mean()
        d["macd"]        = mf - ms
        d["macd_signal"] = d["macd"].ewm(span=self.MACD_SIGNAL_P, adjust=False).mean()
        d["rsi"]         = self._rsi(c, self.RSI_PERIOD)
        d["volume_ma"]   = v.rolling(self.VOLUME_PERIOD).mean()
        d["volume_ok"]   = v > d["volume_ma"]
        d["atr"]         = self._atr(d["high"], d["low"], c, self.ATR_PERIOD)
        return d

    def get_signal(self, df_1h, df_4h, idx):
        row, prev = df_1h.iloc[idx], df_1h.iloc[idx-1]
        price     = float(row["close"])
        regime    = self._get_regime_at(df_4h, df_1h.index[idx])
        if regime == "sideways":
            return self._sig("hold", price, row, regime, "Sideways")
        ema_up   = float(prev["ema_fast"]) <= float(prev["ema_slow"]) and float(row["ema_fast"]) > float(row["ema_slow"])
        ema_down = float(prev["ema_fast"]) >= float(prev["ema_slow"]) and float(row["ema_fast"]) < float(row["ema_slow"])
        macd_up   = float(prev["macd"]) <= float(prev["macd_signal"]) and float(row["macd"]) > float(row["macd_signal"])
        macd_down = float(prev["macd"]) >= float(prev["macd_signal"]) and float(row["macd"]) < float(row["macd_signal"])
        rsi, vol, atr = float(row["rsi"]), bool(row["volume_ok"]), float(row["atr"])
        if ema_up and macd_up and vol and regime == "bullish" and self.RSI_BUY_MIN <= rsi <= self.RSI_BUY_MAX:
            return self._sig("buy", price, row, regime, "3/3", stop=price-atr*self.ATR_STOP, target=price+atr*self.ATR_TARGET)
        if ema_down and macd_down and vol and regime == "bearish" and self.RSI_SELL_MIN <= rsi <= self.RSI_SELL_MAX:
            return self._sig("sell", price, row, regime, "3/3", stop=price+atr*self.ATR_STOP, target=price-atr*self.ATR_TARGET)
        return self._sig("hold", price, row, regime, "Keine Bestaetigung")

    def _get_regime_at(self, df_4h, ts):
        past = df_4h[df_4h.index <= ts]
        return str(past.iloc[-1]["regime"]) if not past.empty and "regime" in past.columns else "neutral"

    def _sig(self, action, price, row, regime, reason, stop=None, target=None):
        return {"action": action, "price": price, "regime": regime, "reason": reason,
                "rsi": float(row.get("rsi", 0)), "macd": float(row.get("macd", 0)),
                "atr": float(row.get("atr", 0)), "ema_fast": float(row.get("ema_fast", 0)),
                "ema_slow": float(row.get("ema_slow", 0)), "volume_ok": bool(row.get("volume_ok", False)),
                "stop": stop, "target": target}

    @staticmethod
    def _rsi(c, p):
        d = c.diff()
        g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
        l = (-d).clip(lower=0).ewm(com=p-1, adjust=False).mean()
        return (100 - 100/(1 + g/l.replace(0, np.nan))).fillna(50)

    @staticmethod
    def _atr(h, l, c, p):
        pc = c.shift(1)
        tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
        return tr.ewm(com=p-1, adjust=False).mean()
