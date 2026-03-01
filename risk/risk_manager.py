"""
risk_manager.py - Sicherheitslayer
Max Trade, Hebel, Drawdown, Pause, Reservepool
"""

from datetime import datetime, timedelta


class RiskManager:
    """
    Regeln:
    - Max 1 Trade gleichzeitig
    - Max Hebel 2x
    - Risiko 2% pro Trade
    - Drawdown > 20% -> 12h Pause
    - 10% jedes Gewinns -> Reservepool
    - Notfall-Stop bei 30% Portfolio Verlust
    """

    MAX_LEVERAGE          = 2.0
    RISK_PER_TRADE        = 0.02
    MAX_DRAWDOWN_PAUSE    = 0.20
    EMERGENCY_STOP        = 0.30
    PAUSE_HOURS           = 12
    RESERVE_RATIO         = 0.10

    def __init__(self, initial_capital: float = 10000):
        self.initial_capital  = initial_capital
        self.capital          = initial_capital
        self.reserve_pool     = 0.0
        self.peak_capital     = initial_capital
        self.open_trades      = []
        self.closed_trades    = []
        self.pause_until      = None
        self.total_wins       = 0
        self.total_losses     = 0
        self.total_pnl        = 0.0

    # ── Pause Check ───────────────────────────────────────────
    def is_paused(self, now: datetime = None) -> bool:
        if self.pause_until is None:
            return False
        now = now or datetime.now()
        if now < self.pause_until:
            return True
        self.pause_until = None
        print(f"[{now}] Resume — Pause beendet")
        return False

    # ── Trade erlaubt? ────────────────────────────────────────
    def check_trade_allowed(self, signal: dict, price: float) -> tuple:
        if self.open_trades:
            return False, "Max 1 Trade aktiv"

        if self.capital < 100:
            return False, "Kapital zu gering"

        drawdown = 1 - (self.capital / self.peak_capital)

        if drawdown >= self.EMERGENCY_STOP:
            return False, f"NOTFALL-STOP: Drawdown {drawdown:.1%}"

        if drawdown >= self.MAX_DRAWDOWN_PAUSE:
            self.pause_until = datetime.now() + timedelta(hours=self.PAUSE_HOURS)
            return False, f"Drawdown {drawdown:.1%} -> {self.PAUSE_HOURS}h Pause"

        if signal.get("stop") is None:
            return False, "Kein Stop definiert"

        return True, "OK"

    # ── Trade öffnen ──────────────────────────────────────────
    def open_trade(self, signal: dict, price: float, ts: datetime = None) -> dict:
        stop_dist   = abs(price - signal["stop"])
        risk_amount = self.capital * self.RISK_PER_TRADE
        position    = (risk_amount / stop_dist) if stop_dist > 0 else 0

        # Hebel begrenzen
        max_pos  = (self.capital * self.MAX_LEVERAGE) / price
        position = min(position, max_pos)

        trade = {
            "id":           len(self.closed_trades) + 1,
            "action":       signal["action"],
            "entry_price":  price,
            "entry_time":   ts or datetime.now(),
            "stop":         signal["stop"],
            "target":       signal["target"],
            "position":     round(position, 6),
            "risk_amount":  round(risk_amount, 2),
            "regime":       signal.get("regime"),
            "rsi":          signal.get("rsi"),
            "atr":          signal.get("atr"),
        }
        self.open_trades.append(trade)
        return trade

    # ── Trade schliessen ──────────────────────────────────────
    def close_trade(self, trade: dict, exit_price: float, ts: datetime = None) -> float:
        direction = 1 if trade["action"] == "buy" else -1
        pnl       = direction * (exit_price - trade["entry_price"]) * trade["position"]

        if pnl > 0:
            reserve        = pnl * self.RESERVE_RATIO
            self.reserve_pool += reserve
            pnl           -= reserve
            self.total_wins += 1
        else:
            self.total_losses += 1

        self.capital   += pnl
        self.total_pnl += pnl

        if self.capital > self.peak_capital:
            self.peak_capital = self.capital

        trade["exit_price"] = exit_price
        trade["exit_time"]  = ts or datetime.now()
        trade["pnl"]        = round(pnl, 2)

        self.open_trades   = [t for t in self.open_trades if t["id"] != trade["id"]]
        self.closed_trades.append(trade)
        return round(pnl, 2)

    # ── Statistiken ───────────────────────────────────────────
    def get_stats(self) -> dict:
        total = self.total_wins + self.total_losses
        wr    = self.total_wins / total if total else 0
        dd    = 1 - (self.capital / self.peak_capital)
        return {
            "Kapital":          round(self.capital, 2),
            "Reserve":          round(self.reserve_pool, 2),
            "Total PnL":        round(self.total_pnl, 2),
            "Return %":         round((self.capital - self.initial_capital) / self.initial_capital * 100, 2),
            "Trades":           total,
            "Wins":             self.total_wins,
            "Losses":           self.total_losses,
            "Win-Rate %":       round(wr * 100, 1),
            "Max Drawdown %":   round(dd * 100, 2),
        }
