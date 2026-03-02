from abc import ABC, abstractmethod
import pandas as pd

class BaseBot(ABC):
    MIN_SCORE      = 8
    MAX_SCORE      = 13
    POINTS_PFLICHT    = 3
    POINTS_EMPFOHLEN  = 2
    POINTS_BONUS      = 1
    ATR_STOP   = 1.5
    ATR_TARGET = 2.5

    def __init__(self, name, regime, capital):
        self.name          = name
        self.target_regime = regime
        self.capital       = capital
        self.open_trade    = None
        self.state         = "SLEEPING"
        self.trades        = []
        self.total_pnl     = 0.0
        self.wins          = 0
        self.losses        = 0

    @abstractmethod
    def compute_indicators(self, df): pass

    @abstractmethod
    def score_signal(self, row, prev, regime): pass

    @abstractmethod
    def calculate_stops(self, row, action): pass

    def activate(self):
        if self.state in ("SLEEPING", "FINISHING"):
            self.state = "SEARCHING" if self.open_trade is None else "IN_TRADE"
            print(f"  [{self.name}] AKTIV")

    def deactivate(self):
        if self.state == "IN_TRADE":
            self.state = "FINISHING"
            print(f"  [{self.name}] FINISHING")
        elif self.state == "SEARCHING":
            self.state = "SLEEPING"
            print(f"  [{self.name}] SLEEPING")

    @property
    def allows_new_trades(self):
        return self.state == "SEARCHING"

    def tick(self, ts, row, prev, regime):
        result = {"bot": self.name, "action": "hold", "state": self.state}
        if self.open_trade:
            outcome = self._manage_trade(row)
            if outcome:
                result.update(outcome)
                return result
        if not self.allows_new_trades:
            return result
        scored = self.score_signal(row, prev, regime)
        result["score"] = scored["score"]
        if scored["score"] >= self.MIN_SCORE and scored["action"] != "hold":
            stop, target = self.calculate_stops(row, scored["action"])
            trade = self._open_trade(ts, scored["action"],
                                     float(row["close"]), stop, target,
                                     scored["reason"], scored["score"])
            result.update({"action": scored["action"], "trade": trade})
        return result

    def _open_trade(self, ts, action, price, stop, target, reason, score):
        stop_dist = abs(price - stop)
        position  = (self.capital * 0.02 / stop_dist) if stop_dist > 0 else 0
        trade = {"id": len(self.trades)+1, "bot": self.name, "action": action,
                 "entry_price": price, "entry_time": ts, "stop": stop,
                 "target": target, "position": round(position, 6),
                 "reason": reason, "score": score}
        self.open_trade = trade
        self.state      = "IN_TRADE"
        print(f"  [{self.name}] {action.upper()} @ {price:.0f} | Stop: {stop:.0f} | Target: {target:.0f} | Score: {score}/13")
        return trade

    def _manage_trade(self, row):
        t    = self.open_trade
        high = float(row["high"])
        low  = float(row["low"])
        hit_target = (t["action"]=="buy" and high>=t["target"]) or (t["action"]=="sell" and low<=t["target"])
        hit_stop   = (t["action"]=="buy" and low<=t["stop"])   or (t["action"]=="sell" and high>=t["stop"])
        if hit_target or hit_stop:
            return self._close_trade(t["target"] if hit_target else t["stop"],
                                     "WIN" if hit_target else "LOSS")
        return None

    def _close_trade(self, exit_price, outcome):
        t   = self.open_trade
        pnl = (1 if t["action"]=="buy" else -1) * (exit_price - t["entry_price"]) * t["position"]
        self.capital   += pnl
        self.total_pnl += pnl
        if outcome == "WIN": self.wins += 1
        else: self.losses += 1
        t.update({"exit_price": exit_price, "pnl": round(pnl,2), "outcome": outcome})
        self.trades.append(t)
        self.open_trade = None
        self.state = "SLEEPING" if self.state == "FINISHING" else "SEARCHING"
        icon = "🟢" if outcome=="WIN" else "🔴"
        print(f"  [{self.name}] {icon} {outcome} @ {exit_price:.0f} | PnL: {pnl:+.0f} | Kapital: {self.capital:.0f}")
        return {"action": "close", "outcome": outcome, "pnl": pnl, "trade": t}

    def get_stats(self):
        total = self.wins + self.losses
        return {"Bot": self.name, "State": self.state, "Trades": total,
                "Wins": self.wins, "Losses": self.losses,
                "Win-Rate %": round(self.wins/total*100 if total else 0, 1),
                "PnL USDT": round(self.total_pnl, 2), "Kapital": round(self.capital, 2)}
