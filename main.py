"""
main.py - Trading Agent Orchestrator
Verbindet alle Module und startet Backtest
"""

import sys
sys.path.insert(0, '/docker/trading-agent')

from datetime import datetime
from data.feed import fetch_multi_timeframe
from modules.signal_generator import SignalGenerator
from modules.logger import TradeLogger
from risk.risk_manager import RiskManager

# ── Konfiguration ─────────────────────────────────────────────
CONFIG = {
    "symbol":          "BTC/USDT",
    "initial_capital": 10000,
    "log_dir":         "logs",
}

def run():
    print("=" * 50)
    print(f"  Trading Agent Phase 1")
    print(f"  {CONFIG['symbol']} | Kapital: {CONFIG['initial_capital']} USDT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 50)

    # Module initialisieren
    sg     = SignalGenerator()
    rm     = RiskManager(CONFIG["initial_capital"])
    logger = TradeLogger(CONFIG["log_dir"])

    # Daten laden
    print("\n📡 Lade Marktdaten...")
    data  = fetch_multi_timeframe(CONFIG["symbol"])
    df_4h = sg.compute_regime(data["4h"])
    df_1h = sg.compute_signals(data["1h"])
    print(f"✅ Regime aktuell: {df_4h['regime'].iloc[-1].upper()}")

    open_trade = None
    signals_found = 0
    start_idx = max(sg.EMA_MACRO, sg.ICHIMOKU_SPAN_B + sg.ICHIMOKU_BASE) + 10

    print(f"\n🔍 Scanne {len(df_1h) - start_idx} Kerzen...\n")

    for i in range(start_idx, len(df_1h)):
        row   = df_1h.iloc[i]
        ts    = df_1h.index[i]
        price = float(row["close"])

        # Pause Check
        if rm.is_paused(ts.to_pydatetime()):
            continue

        # Offenen Trade verwalten
        if open_trade:
            high = float(row["high"])
            low  = float(row["low"])
            action = open_trade["action"]

            hit_stop   = (action == "buy"  and low  <= open_trade["stop"]) or \
                         (action == "sell" and high >= open_trade["stop"])
            hit_target = (action == "buy"  and high >= open_trade["target"]) or \
                         (action == "sell" and low  <= open_trade["target"])

            if hit_target or hit_stop:
                exit_price = open_trade["target"] if hit_target else open_trade["stop"]
                outcome    = "WIN" if hit_target else "LOSS"
                pnl        = rm.close_trade(open_trade, exit_price, ts.to_pydatetime())
                logger.log_trade_close(ts, open_trade, exit_price, pnl, outcome)
                icon = "🟢" if hit_target else "🔴"
                print(f"  {icon} {outcome} @ {exit_price:.0f} | PnL: {pnl:+.0f} USDT | "
                      f"Kapital: {rm.capital:.0f}")
                open_trade = None
            continue

        # Signal prüfen
        signal = sg.get_signal(df_1h, df_4h, i)

        if signal["action"] in ("buy", "sell"):
            allowed, reason = rm.check_trade_allowed(signal, price)
            if not allowed:
                continue

            open_trade = rm.open_trade(signal, price, ts.to_pydatetime())
            logger.log_signal(ts, signal, open_trade)
            signals_found += 1

            print(f"  {'🔵' if signal['action'] == 'buy' else '🟠'} "
                  f"{signal['action'].upper()} @ {price:.0f} | "
                  f"Stop: {open_trade['stop']:.0f} | "
                  f"Target: {open_trade['target']:.0f} | "
                  f"RSI: {signal['rsi']:.1f} | "
                  f"Regime: {signal['regime']}")

    # Zusammenfassung
    stats = rm.get_stats()
    logger.save_summary(stats)

    print("\n" + "=" * 50)
    print("  BACKTEST ERGEBNIS")
    print("=" * 50)
    for k, v in stats.items():
        print(f"  {k:<20} {v}")
    print("=" * 50)
    print(f"\n📁 Logs gespeichert in: {CONFIG['log_dir']}/")


if __name__ == "__main__":
    run()
