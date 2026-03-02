import sys
sys.path.insert(0, '/docker/trading-agent')
import pandas as pd
from data.feed import fetch_multi_timeframe
from core.regime_detector import RegimeDetector
from bots.trend_rider import TrendRider
from bots.band_bouncer import BandBouncer
from bots.div_hunter import DivHunter
from modules.logger import TradeLogger

class Orchestrator:
    SWITCH_COOLDOWN = 3  # Kerzen warten nach Switch

    def __init__(self, capital=10000, log_dir="logs"):
        third = round(capital / 3, 2)
        self.regime_detector = RegimeDetector()
        self.bots = {
            "bullish":  TrendRider(third),
            "sideways": BandBouncer(third),
            "bearish":  DivHunter(third),
        }
        self.active_regime  = None
        self.logger         = TradeLogger(log_dir)
        self.capital        = capital
        self.switches       = 0
        self.cooldown_left  = 0

    def _activate(self, regime):
        if regime == self.active_regime:
            return
        print(f"\n🔀 SWITCH: {self.active_regime} → {regime}")
        if self.active_regime and self.active_regime in self.bots:
            self.bots[self.active_regime].deactivate()
        if regime in self.bots:
            self.bots[regime].activate()
        self.active_regime = regime
        self.cooldown_left = self.SWITCH_COOLDOWN
        self.switches += 1

    def run(self, symbol="BTC/USDT"):
        print("=" * 55)
        print("  Trading Agent — 3-Spur System")
        print(f"  Kapital: {self.capital} USDT (je Spur: {round(self.capital/3)})")
        print(f"  Cooldown nach Switch: {self.SWITCH_COOLDOWN} Kerzen")
        print("=" * 55)

        data  = fetch_multi_timeframe(symbol)
        df_4h = self.regime_detector.compute_all(data["4h"])

        df = {}
        for regime, bot in self.bots.items():
            df[regime] = bot.compute_indicators(data["1h"])

        start = 210
        print(f"\n🔍 Scanne {len(df['bullish']) - start} Kerzen...\n")

        for i in range(start, len(df["bullish"])):
            ts = df["bullish"].index[i]

            # Regime Update
            past_4h = df_4h[df_4h.index <= ts]
            if not past_4h.empty:
                new_regime = past_4h.iloc[-1]["regime"]
                self._activate(new_regime)

            # Cooldown zählen
            if self.cooldown_left > 0:
                self.cooldown_left -= 1
                # Alle Bots ticken aber aktiver Bot darf keine neuen Trades
                for regime, bot in self.bots.items():
                    if bot.open_trade:
                        row  = df[regime].iloc[i]
                        prev = df[regime].iloc[i-1]
                        bot.tick(ts, row, prev, "cooldown")
                continue

            # Alle 3 Bots ticken
            for regime, bot in self.bots.items():
                row  = df[regime].iloc[i]
                prev = df[regime].iloc[i-1]
                bot.tick(ts, row, prev, self.active_regime or "neutral")

        self._print_results()

    def _print_results(self):
        print("\n" + "=" * 55)
        print("  3-SPUR BACKTEST ERGEBNIS")
        print("=" * 55)
        print(f"  Regime Switches: {self.switches}")
        print(f"  Cooldown:        {self.SWITCH_COOLDOWN} Kerzen")
        print()
        total_pnl = 0
        for regime, bot in self.bots.items():
            s = bot.get_stats()
            icon = "🟢" if regime=="bullish" else "🟡" if regime=="sideways" else "🔴"
            print(f"  {icon} {s['Bot']:<14} | Trades: {s['Trades']:>3} | "
                  f"WR: {s['Win-Rate %']:>5}% | PnL: {s['PnL USDT']:>+8.2f} USDT")
            total_pnl += s["PnL USDT"]
        print(f"\n  {'GESAMT':<20} | PnL: {total_pnl:>+8.2f} USDT")
        ret = total_pnl / self.capital * 100
        print(f"  {'Return':<20} | {ret:>+.2f}%")
        print("=" * 55)

if __name__ == "__main__":
    o = Orchestrator(10000)
    o.run()
