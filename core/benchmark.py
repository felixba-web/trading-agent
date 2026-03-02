import sys
sys.path.insert(0, '/docker/trading-agent')
import pandas as pd
from data.feed import fetch_multi_timeframe
from core.regime_detector import RegimeDetector
from core.orchestrator import Orchestrator
from bots.trend_rider import TrendRider
from bots.band_bouncer import BandBouncer
from bots.div_hunter import DivHunter

CAPITAL = 10000

def run_solo(bot, df_1h, df_4h, regime_detector, name):
    """Bot läuft solo — immer aktiv, ignoriert Regime"""
    rd = RegimeDetector()
    df_4h2 = rd.compute_all(df_4h.copy())
    df = bot.compute_indicators(df_1h.copy())
    start = 210
    for i in range(start, len(df)):
        ts      = df.index[i]
        row     = df.iloc[i]
        prev    = df.iloc[i-1]
        past_4h = df_4h2[df_4h2.index <= ts]
        regime  = past_4h.iloc[-1]["regime"] if not past_4h.empty else "neutral"
        if bot.state == "SLEEPING":
            bot.activate()
        bot.tick(ts, row, prev, regime)
    return bot.get_stats()

def run_buyhold(df_1h, capital):
    """Buy & Hold Benchmark"""
    start_price = float(df_1h["close"].iloc[210])
    end_price   = float(df_1h["close"].iloc[-1])
    pnl         = (end_price - start_price) / start_price * capital
    ret         = (end_price - start_price) / start_price * 100
    return {
        "Bot": "BuyHold BTC",
        "Trades": 1,
        "Win-Rate %": 100.0 if pnl > 0 else 0.0,
        "PnL USDT": round(pnl, 2),
        "Return %": round(ret, 2),
        "Start": round(start_price, 0),
        "End":   round(end_price, 0),
    }

def run_3spur(capital):
    """3-Spur System"""
    o = Orchestrator(capital)
    # suppress prints
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        o.run()
    results = []
    total_pnl = 0
    for regime, bot in o.bots.items():
        s = bot.get_stats()
        total_pnl += s["PnL USDT"]
        results.append(s)
    return results, total_pnl

def print_table(results):
    print("\n" + "=" * 70)
    print(f"  {'Bot':<16} | {'Trades':>6} | {'WR %':>6} | {'PnL USDT':>10} | {'Return %':>8}")
    print("=" * 70)
    for r in results:
        ret = r.get("Return %", round(r.get("PnL USDT", 0) / CAPITAL * 100, 2))
        icon = "🟢" if r.get("PnL USDT", 0) > 0 else "🔴"
        print(f"  {icon} {r['Bot']:<14} | {r.get('Trades', 0):>6} | "
              f"{r.get('Win-Rate %', 0):>5.1f}% | "
              f"{r.get('PnL USDT', 0):>+10.2f} | {ret:>+7.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    print("=" * 70)
    print("  BENCHMARK — 3-Spur vs Solo vs Buy&Hold")
    print("=" * 70)

    data = fetch_multi_timeframe()
    df_1h = data["1h"]
    df_4h = data["4h"]

    print("\n📊 Starte Solo-Tests...\n")

    rd = RegimeDetector()
    df_4h_reg = rd.compute_all(df_4h.copy())

    tr_stats = run_solo(TrendRider(CAPITAL), df_1h, df_4h, rd, "TrendRider Solo")
    bb_stats = run_solo(BandBouncer(CAPITAL), df_1h, df_4h, rd, "BandBouncer Solo")
    dh_stats = run_solo(DivHunter(CAPITAL), df_1h, df_4h, rd, "DivHunter Solo")

    print("\n📊 Starte 3-Spur System...\n")
    spur_results, spur_pnl = run_3spur(CAPITAL)

    bh = run_buyhold(df_1h, CAPITAL)

    # Ergebnis
    all_results = [
        {"Bot": "─── SOLO ───", "Trades": "", "Win-Rate %": 0, "PnL USDT": 0},
        tr_stats,
        bb_stats,
        dh_stats,
        {"Bot": "─── 3-SPUR ──", "Trades": "", "Win-Rate %": 0, "PnL USDT": 0},
    ] + spur_results + [
        {"Bot": "3-Spur GESAMT", "Trades": sum(s["Trades"] for s in spur_results),
         "Win-Rate %": round(sum(s["Wins"] for s in spur_results) /
                             max(sum(s["Trades"] for s in spur_results), 1) * 100, 1),
         "PnL USDT": round(spur_pnl, 2)},
        {"Bot": "─── BENCHMARK", "Trades": "", "Win-Rate %": 0, "PnL USDT": 0},
        bh,
    ]

    print_table(all_results)

    # Gewinner
    print(f"\n  BTC Start:    ${bh['Start']:,.0f}")
    print(f"  BTC End:      ${bh['End']:,.0f}")
    print(f"  Buy&Hold:     {bh['Return %']:+.2f}%")
    print(f"  3-Spur:       {round(spur_pnl/CAPITAL*100, 2):+.2f}%")
    diff = round(spur_pnl/CAPITAL*100 - bh['Return %'], 2)
    icon = "✅" if diff > 0 else "❌"
    print(f"  Differenz:    {diff:+.2f}% {icon}")
    print("=" * 70)
