import pandas as pd

class RegimeDetector:
    CONVERSION_PERIOD = 9
    BASE_PERIOD       = 26
    SPAN_B_PERIOD     = 52
    CONFIRM_STANDARD  = 3
    CONFIRM_THIN_KUMO = 5
    THIN_KUMO_PCT     = 0.5

    def __init__(self):
        self._confirmed_regime = "sideways"
        self._pending_regime   = None
        self._pending_count    = 0

    def compute_kumo(self, df):
        h, l = df["high"], df["low"]
        conv   = (h.rolling(self.CONVERSION_PERIOD).max() + l.rolling(self.CONVERSION_PERIOD).min()) / 2
        base   = (h.rolling(self.BASE_PERIOD).max() + l.rolling(self.BASE_PERIOD).min()) / 2
        span_a = ((conv + base) / 2).shift(self.BASE_PERIOD)
        span_b = ((h.rolling(self.SPAN_B_PERIOD).max() + l.rolling(self.SPAN_B_PERIOD).min()) / 2).shift(self.BASE_PERIOD)
        d = df.copy()
        d["kumo_top"]       = pd.concat([span_a, span_b], axis=1).max(axis=1)
        d["kumo_bottom"]    = pd.concat([span_a, span_b], axis=1).min(axis=1)
        d["kumo_width_pct"] = ((d["kumo_top"] - d["kumo_bottom"]) / d["close"] * 100).round(3)
        return d

    def _raw_regime(self, row):
        if pd.isna(row["kumo_top"]): return "neutral"
        if row["close"] > row["kumo_top"]: return "bullish"
        if row["close"] < row["kumo_bottom"]: return "bearish"
        return "sideways"

    def _required_candles(self, w):
        return self.CONFIRM_THIN_KUMO if w < self.THIN_KUMO_PCT else self.CONFIRM_STANDARD

    def update(self, row):
        raw      = self._raw_regime(row)
        width    = float(row.get("kumo_width_pct", 1.0))
        required = self._required_candles(width)
        switched = False
        if raw == self._confirmed_regime:
            self._pending_regime = None
            self._pending_count  = 0
        elif raw == self._pending_regime:
            self._pending_count += 1
            if self._pending_count >= required:
                self._confirmed_regime = self._pending_regime
                self._pending_regime   = None
                self._pending_count    = 0
                switched               = True
        else:
            self._pending_regime = raw
            self._pending_count  = 1
        return {"regime": self._confirmed_regime, "switched": switched,
                "pending": self._pending_regime, "count": self._pending_count,
                "required": required, "kumo_width": width}

    def get_regime(self):
        return self._confirmed_regime

    def compute_all(self, df):
        df = self.compute_kumo(df)
        regimes, switched = [], []
        self._confirmed_regime = "sideways"
        self._pending_regime   = None
        self._pending_count    = 0
        for i in range(len(df)):
            r = self.update(df.iloc[i])
            regimes.append(r["regime"])
            switched.append(r["switched"])
        df["regime"]   = regimes
        df["switched"] = switched
        return df
