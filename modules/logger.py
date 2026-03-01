"""
logger.py - Trade & Signal Logging
CSV + Excel Export
"""

import os
import pandas as pd
from datetime import datetime


class TradeLogger:

    def __init__(self, log_dir: str = "logs"):
        self.log_dir  = log_dir
        os.makedirs(log_dir, exist_ok=True)
        ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = os.path.join(log_dir, f"trades_{ts}.csv")
        self.xlsx_file= os.path.join(log_dir, f"trades_{ts}.xlsx")
        self.sum_file = os.path.join(log_dir, f"summary_{ts}.csv")
        self._trades  = []
        self._signals = []

    def log_signal(self, ts, signal: dict, trade: dict):
        self._signals.append({
            "timestamp":    ts,
            "action":       signal["action"],
            "price":        round(signal["price"], 2),
            "regime":       signal.get("regime"),
            "reason":       signal.get("reason"),
            "rsi":          round(signal.get("rsi", 0), 2),
            "macd":         round(signal.get("macd", 0), 4),
            "atr":          round(signal.get("atr", 0), 2),
            "ema_fast":     round(signal.get("ema_fast", 0), 2),
            "ema_slow":     round(signal.get("ema_slow", 0), 2),
            "volume_ok":    signal.get("volume_ok"),
            "stop":         round(trade["stop"], 2),
            "target":       round(trade["target"], 2),
            "position":     trade.get("position"),
            "risk_usdt":    trade.get("risk_amount"),
        })

    def log_trade_close(self, ts, trade: dict, exit_price: float,
                        pnl: float, outcome: str):
        self._trades.append({
            "trade_id":     trade["id"],
            "action":       trade["action"],
            "entry_time":   trade.get("entry_time"),
            "exit_time":    ts,
            "entry_price":  trade["entry_price"],
            "exit_price":   round(exit_price, 2),
            "stop":         round(trade["stop"], 2),
            "target":       round(trade["target"], 2),
            "position":     trade.get("position"),
            "pnl_usdt":     round(pnl, 2),
            "outcome":      outcome,
            "regime":       trade.get("regime"),
            "rsi_entry":    trade.get("rsi"),
            "atr_entry":    trade.get("atr"),
        })
        self._flush()

    def save_summary(self, stats: dict):
        pd.DataFrame([stats]).to_csv(self.sum_file, index=False)
        print(f"Summary: {self.sum_file}")
        self._export_excel()

    def _flush(self):
        if self._trades:
            pd.DataFrame(self._trades).to_csv(self.csv_file, index=False)

    def _export_excel(self):
        try:
            with pd.ExcelWriter(self.xlsx_file, engine="openpyxl") as w:
                if self._trades:
                    pd.DataFrame(self._trades).to_excel(w, sheet_name="Trades", index=False)
                if self._signals:
                    pd.DataFrame(self._signals).to_excel(w, sheet_name="Signale", index=False)
            print(f"Excel: {self.xlsx_file}")
        except Exception as e:
            print(f"Excel Fehler: {e}")
