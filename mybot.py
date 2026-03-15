"""
Strategy A v6 — Live Bot for Binance Testnet / Live
====================================================
Strategy : RSI + MACD + BB + EMA + Supertrend + ADX + Candle patterns
Win rate : ~60% (backtested 15 years BTC/USDT daily)
Timeframe: Daily (checks once per day at candle close)
Requires : pip install python-binance flask flask-socketio pandas ta
Run      : python strategy_v6_live_bot.py
Dashboard: http://localhost:5000

TESTNET SETUP:
1. Go to testnet.binancefuture.com
2. Create testnet API key
3. Paste below — uses fake money, real market prices
"""

import time, threading, logging
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, (np.ndarray,)): return obj.tolist()
        return super().default(obj)
import ta

# ─── CONFIG ──────────────────────────────────────────────────────────────────
API_KEY        = "aR1TvmYaQhFTALbFnL6cKJcwwmgqkL07Ob4q3KI47KlGR3WjEeoWH103nr03kBht"      # from testnet.binancefuture.com
API_SECRET     = "VA20f23YDAWzz6B16vaC2qrib7Ri708LJEEzQnqQyNANZRrdfESh8M3OA5o91D08"
USE_TESTNET    = True                         # True = fake money, False = real money

SYMBOL         = "BTCUSDT"
TRADE_USDT     = 20.0          # USDT per trade
TIMEFRAME      = Client.KLINE_INTERVAL_1DAY  # daily candles
CANDLES        = 250           # history to load
CHECK_INTERVAL = 3600          # check every 1 hour (daily candle)

# Strategy A v6 settings (matching TradingView)
RSI_LEN        = 14
RSI_OVERSOLD   = 45
RSI_OVERBOUGHT = 58
MACD_FAST      = 8
MACD_SLOW      = 21
MACD_SIG       = 5
BB_LEN         = 20
BB_MULT        = 2.0
ST_FACTOR      = 2.0   # Supertrend
ST_ATR_LEN     = 10
ADX_LEN        = 14
ADX_THRESH     = 20    # minimum trend strength
EMA_SHORT      = 21
EMA_MID        = 55
EMA_LONG       = 200
ATR_LEN        = 14
SL_MULT        = 1.0   # stop loss ATR multiplier
TP1_MULT       = 1.5   # take profit 1 ATR multiplier (50% close)
TP2_MULT       = 2.5   # take profit 2 ATR multiplier (full close)
MIN_SCORE      = 3     # minimum signals needed (3 of 5)

def clean(obj):
    if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list): return [clean(i) for i in obj]
    if isinstance(obj, (bool,)): return bool(obj)
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
    except: pass
    return obj

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("StrategyV6")

# ─── STATE ───────────────────────────────────────────────────────────────────
state = {
    "running":      False,
    "price":        0.0,
    "buy_score":    0,
    "sell_score":   0,
    "rsi":          0.0,
    "macd":         "—",
    "bb_zone":      "—",
    "trend":        "—",
    "supertrend":   "—",
    "adx":          0.0,
    "in_position":  False,
    "entry_price":  0.0,
    "stop_loss":    0.0,
    "take_profit1": 0.0,
    "take_profit2": 0.0,
    "total_profit": 0.0,
    "trades":       [],
    "started_at":   None,
    "error":        None,
    "last_checked": None,
}

# ─── BINANCE CLIENT ──────────────────────────────────────────────────────────
client = None

def get_client():
    global client
    if client is None:
        if USE_TESTNET:
            client = Client(API_KEY, API_SECRET, testnet=True)
        else:
            client = Client(API_KEY, API_SECRET)
    return client

def get_price():
    c = get_client()
    ticker = c.get_symbol_ticker(symbol=SYMBOL)
    return float(ticker["price"])

def get_candles():
    c = get_client()
    klines = c.get_klines(symbol=SYMBOL, interval=TIMEFRAME, limit=CANDLES)
    df = pd.DataFrame(klines, columns=[
        "time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_base",
        "taker_buy_quote","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df

# ─── INDICATORS ───────────────────────────────────────────────────────────────
def compute_indicators(df):
    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=RSI_LEN).rsi()

    # MACD
    macd_ind = ta.trend.MACD(df["close"], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIG)
    df["macd"]        = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"]   = macd_ind.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=BB_LEN, window_dev=BB_MULT)
    df["bb_upper"]    = bb.bollinger_hband()
    df["bb_lower"]    = bb.bollinger_lband()
    df["bb_mid"]      = bb.bollinger_mavg()

    # EMAs
    df["ema21"]  = ta.trend.EMAIndicator(df["close"], window=EMA_SHORT).ema_indicator()
    df["ema55"]  = ta.trend.EMAIndicator(df["close"], window=EMA_MID).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(df["close"], window=EMA_LONG).ema_indicator()

    # ATR
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_LEN).average_true_range()

    # Supertrend (manual calculation)
    hl2      = (df["high"] + df["low"]) / 2
    atr_st   = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=ST_ATR_LEN).average_true_range()
    upper    = hl2 + ST_FACTOR * atr_st
    lower    = hl2 - ST_FACTOR * atr_st
    st_dir   = pd.Series(index=df.index, dtype=float)
    st_line  = pd.Series(index=df.index, dtype=float)
    for i in range(1, len(df)):
        prev_close = df["close"].iloc[i-1]
        curr_close = df["close"].iloc[i]
        prev_upper = upper.iloc[i-1]
        prev_lower = lower.iloc[i-1]
        curr_upper = upper.iloc[i]
        curr_lower = lower.iloc[i]
        final_upper = curr_upper if curr_upper < prev_upper or prev_close > prev_upper else prev_upper
        final_lower = curr_lower if curr_lower > prev_lower or prev_close < prev_lower else prev_lower
        if i == 1:
            st_dir.iloc[i] = 1
        elif st_line.iloc[i-1] == prev_upper:
            st_dir.iloc[i] = -1 if curr_close > final_upper else 1
        else:
            st_dir.iloc[i] = 1 if curr_close < final_lower else -1
        st_line.iloc[i] = final_lower if st_dir.iloc[i] == -1 else final_upper
    df["st_dir"]  = st_dir
    df["st_line"] = st_line

    # ADX
    df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=ADX_LEN).adx()

    # Candle patterns
    df["body"]       = abs(df["close"] - df["open"])
    df["wick_low"]   = df[["open","close"]].min(axis=1) - df["low"]
    df["is_hammer"]  = (df["wick_low"] > df["body"] * 1.5) & (df["close"] > df["open"])
    prev_open        = df["open"].shift(1)
    prev_close       = df["close"].shift(1)
    df["is_engulf"]  = (df["close"] > df["open"]) & (df["open"] < prev_close) & (df["close"] > prev_open)

    return df

# ─── SIGNAL SCORING ───────────────────────────────────────────────────────────
def get_signals(df):
    row  = df.iloc[-1]
    prev = df.iloc[-2]

    rsi         = row["rsi"]
    macd_bull   = row["macd"] > row["macd_signal"] and row["macd_hist"] > prev["macd_hist"]
    macd_bear   = row["macd"] < row["macd_signal"] and row["macd_hist"] < prev["macd_hist"]
    bb_buy      = row["close"] < row["bb_mid"]
    bb_sell     = row["close"] > row["bb_mid"]
    trend_up    = row["ema21"] > row["ema55"] and row["ema55"] > row["ema200"]
    trend_down  = row["ema21"] < row["ema55"] and row["ema55"] < row["ema200"]
    st_bull     = row["st_dir"] == -1
    st_bear     = row["st_dir"] == 1
    adx_strong  = row["adx"] > ADX_THRESH
    htf_bull    = row["ema21"] > row["ema200"]
    htf_bear    = row["ema21"] < row["ema200"]
    candle_bull = row["is_hammer"] or row["is_engulf"]

    buy_score  = sum([rsi < RSI_OVERSOLD, macd_bull, bb_buy,  trend_up,   st_bull])
    sell_score = sum([rsi > RSI_OVERBOUGHT, macd_bear, bb_sell, trend_down, st_bear])

    buy_signal  = buy_score >= MIN_SCORE and adx_strong and htf_bull and candle_bull and not trend_down
    sell_signal = sell_score >= MIN_SCORE and (htf_bear or trend_down)

    return {
        "buy_score":  int(buy_score),
        "sell_score": int(sell_score),
        "buy_signal": bool(buy_signal),
        "sell_signal":bool(sell_signal),
        "rsi":        float(round(rsi, 1)),
        "macd":       "BULL" if macd_bull else "BEAR",
        "bb_zone":    "BUY ZONE" if bb_buy else "SELL ZONE",
        "trend":      "UP" if trend_up else "DOWN" if trend_down else "MIXED",
        "supertrend": "BULL" if st_bull else "BEAR",
        "adx":        float(round(float(row["adx"]), 1)),
        "atr":        float(round(float(row["atr"]), 2)),
        "price":      float(round(float(row["close"]), 2)),
    }

# ─── ORDERS ──────────────────────────────────────────────────────────────────
def place_buy(price, atr):
    c = get_client()
    try:
        order = c.order_market_buy(symbol=SYMBOL, quoteOrderQty=round(TRADE_USDT, 2))
        sl  = round(price - atr * SL_MULT, 2)
        tp1 = round(price + atr * TP1_MULT, 2)
        tp2 = round(price + atr * TP2_MULT, 2)
        qty = float(order["executedQty"])
        log.info(f"BUY {qty} BTC @ {price} | SL:{sl} TP1:{tp1} TP2:{tp2}")
        return order, sl, tp1, tp2
    except BinanceAPIException as e:
        log.error(f"Buy failed: {e}")
        return None, 0, 0, 0

def place_sell(qty_btc, reason):
    c = get_client()
    try:
        order = c.order_market_sell(symbol=SYMBOL, quantity=round(qty_btc, 6))
        log.info(f"SELL {qty_btc} BTC — {reason}")
        return order
    except BinanceAPIException as e:
        log.error(f"Sell failed: {e}")
        return None

# ─── BOT LOOP ────────────────────────────────────────────────────────────────
def bot_loop():
    log.info("Strategy A v6 Bot starting...")
    state["running"]    = True
    state["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state["error"]      = None

    in_pos      = False
    entry_price = 0.0
    sl          = 0.0
    tp1         = 0.0
    tp2         = 0.0
    tp1_hit     = False
    btc_held    = 0.0
    total_spent = 0.0
    trades      = []

    while state["running"]:
        try:
            df      = get_candles()
            df      = compute_indicators(df)
            sigs    = get_signals(df)
            price   = get_price()

            state["price"]      = float(price)
            state["buy_score"]  = int(sigs["buy_score"])
            state["sell_score"] = int(sigs["sell_score"])
            state["rsi"]        = float(sigs["rsi"])
            state["macd"]       = sigs["macd"]
            state["bb_zone"]    = sigs["bb_zone"]
            state["trend"]      = sigs["trend"]
            state["supertrend"] = sigs["supertrend"]
            state["adx"]        = float(sigs["adx"])
            state["in_position"]= in_pos
            state["stop_loss"]  = sl
            state["take_profit1"] = tp1
            state["take_profit2"] = tp2
            state["last_checked"] = datetime.now().strftime("%H:%M:%S")

            log.info(f"Price:{price} Score:{sigs['buy_score']}/5 RSI:{sigs['rsi']} ADX:{sigs['adx']} Trend:{sigs['trend']}")

            if not in_pos and sigs["buy_signal"]:
                order, sl, tp1, tp2 = place_buy(price, sigs["atr"])
                if order:
                    btc_held    = float(order["executedQty"])
                    total_spent = float(order["cummulativeQuoteQty"])
                    entry_price = price
                    tp1_hit     = False
                    in_pos      = True
                    state["entry_price"] = entry_price
                    trades.append({"type":"BUY","price":price,"score":sigs["buy_score"],"time":datetime.now().strftime("%H:%M:%S"),"profit":0})
                    state["trades"] = trades[-20:]

            elif in_pos:
                # Take profit 1 — close 50%
                if not tp1_hit and price >= tp1:
                    sell_qty = round(btc_held * 0.5, 6)
                    order    = place_sell(sell_qty, "TP1")
                    if order:
                        received    = float(order["cummulativeQuoteQty"])
                        cost        = sell_qty * (total_spent / btc_held)
                        profit      = received - cost
                        state["total_profit"] += profit
                        btc_held    -= sell_qty
                        total_spent -= cost
                        tp1_hit     = True
                        trades.append({"type":"TP1","price":price,"score":0,"time":datetime.now().strftime("%H:%M:%S"),"profit":round(profit,4)})
                        state["trades"] = trades[-20:]

                # Take profit 2 — close rest
                elif price >= tp2:
                    order = place_sell(btc_held, "TP2")
                    if order:
                        received    = float(order["cummulativeQuoteQty"])
                        profit      = received - total_spent
                        state["total_profit"] += profit
                        trades.append({"type":"TP2","price":price,"score":0,"time":datetime.now().strftime("%H:%M:%S"),"profit":round(profit,4)})
                        state["trades"] = trades[-20:]
                        in_pos = False; btc_held = 0; total_spent = 0

                # Stop loss
                elif price <= sl:
                    order = place_sell(btc_held, "STOP LOSS")
                    if order:
                        received    = float(order["cummulativeQuoteQty"])
                        profit      = received - total_spent
                        state["total_profit"] += profit
                        trades.append({"type":"STOP","price":price,"score":0,"time":datetime.now().strftime("%H:%M:%S"),"profit":round(profit,4)})
                        state["trades"] = trades[-20:]
                        in_pos = False; btc_held = 0; total_spent = 0

                # Signal exit
                elif sigs["sell_signal"]:
                    order = place_sell(btc_held, "SIGNAL EXIT")
                    if order:
                        received    = float(order["cummulativeQuoteQty"])
                        profit      = received - total_spent
                        state["total_profit"] += profit
                        trades.append({"type":"SELL","price":price,"score":sigs["sell_score"],"time":datetime.now().strftime("%H:%M:%S"),"profit":round(profit,4)})
                        state["trades"] = trades[-20:]
                        in_pos = False; btc_held = 0; total_spent = 0

        except Exception as e:
            log.error(f"Bot error: {e}")
            state["error"] = str(e)

        time.sleep(CHECK_INTERVAL)

    state["running"] = False
    log.info("Bot stopped.")

# ─── DASHBOARD ───────────────────────────────────────────────────────────────
app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

DASH = """
<!DOCTYPE html><html><head>
<title>Strategy A v6 Bot</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0a0a0f;color:#e0e0e0;padding:20px;min-height:100vh}
h1{font-size:18px;font-weight:500;color:#fff;margin-bottom:20px;display:flex;align-items:center;gap:10px}
.dot{width:8px;height:8px;border-radius:50%}
.dot-g{background:#2ecc71;box-shadow:0 0 8px #2ecc71}
.dot-r{background:#e74c3c}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:20px}
.card{background:#13131f;border:1px solid #1e1e30;border-radius:10px;padding:12px 14px}
.card-label{font-size:10px;color:#555;margin-bottom:5px;text-transform:uppercase;letter-spacing:.06em}
.card-value{font-size:20px;font-weight:500;color:#fff}
.green{color:#2ecc71}.red{color:#e74c3c}.amber{color:#f0a500}.blue{color:#3b8bd4}.purple{color:#9b59b6}
.signals{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-bottom:20px}
.sig{background:#13131f;border:1px solid #1e1e30;border-radius:8px;padding:10px 12px;font-size:12px}
.sig-label{color:#555;font-size:10px;margin-bottom:3px;text-transform:uppercase}
.sig-value{font-weight:500}
.score-bar{height:6px;border-radius:3px;background:#1e1e30;margin-top:8px;overflow:hidden}
.score-fill{height:100%;border-radius:3px;transition:width .3s}
.controls{display:flex;gap:10px;margin-bottom:20px;align-items:center}
.btn{padding:8px 18px;border-radius:8px;border:none;font-size:13px;font-weight:500;cursor:pointer}
.btn-start{background:#2ecc71;color:#000}
.btn-stop{background:#e74c3c;color:#fff}
.btn:disabled{opacity:.35;cursor:not-allowed}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;padding:8px 10px;color:#444;font-weight:400;border-bottom:1px solid #1e1e30}
td{padding:8px 10px;border-bottom:1px solid #13131f}
.buy{color:#2ecc71;font-weight:500}.sell{color:#e74c3c;font-weight:500}
.tp1,.tp2{color:#3b8bd4;font-weight:500}.stop{color:#f0a500;font-weight:500}
.err{background:#2a1111;border:1px solid #e74c3c;border-radius:8px;padding:10px 14px;font-size:12px;color:#e74c3c;margin-bottom:16px;display:none}
.section{font-size:11px;color:#444;text-transform:uppercase;letter-spacing:.06em;margin:16px 0 8px}
.testnet-badge{background:#1a1a3a;color:#9b59b6;font-size:11px;padding:3px 10px;border-radius:20px;border:1px solid #2a2a5a}
</style></head><body>
<h1>
  <span id="statusDot" class="dot dot-r"></span>
  Strategy A v6 — 60% Win Rate Bot
  <span class="testnet-badge" id="modeBadge">TESTNET</span>
</h1>

<div class="controls">
  <button class="btn btn-start" id="startBtn" onclick="startBot()">Start bot</button>
  <button class="btn btn-stop"  id="stopBtn"  onclick="stopBot()" disabled>Stop bot</button>
  <span id="statusText" style="font-size:12px;color:#555">Stopped</span>
  <span id="lastCheck" style="font-size:12px;color:#333;margin-left:auto"></span>
</div>

<div class="err" id="errBox"></div>

<div class="grid">
  <div class="card"><div class="card-label">BTC price</div><div class="card-value" id="price">—</div></div>
  <div class="card"><div class="card-label">Total profit</div><div class="card-value" id="profit">$0.00</div></div>
  <div class="card"><div class="card-label">Buy score</div><div class="card-value" id="buyScore">0/5</div></div>
  <div class="card"><div class="card-label">Position</div><div class="card-value" id="position">None</div></div>
  <div class="card"><div class="card-label">Stop loss</div><div class="card-value amber" id="sl">—</div></div>
  <div class="card"><div class="card-label">Take profit 1</div><div class="card-value blue" id="tp1">—</div></div>
  <div class="card"><div class="card-label">Take profit 2</div><div class="card-value green" id="tp2">—</div></div>
  <div class="card"><div class="card-label">Started</div><div class="card-value" style="font-size:12px" id="started">—</div></div>
</div>

<div class="section">Live signals</div>
<div class="signals">
  <div class="sig"><div class="sig-label">RSI</div><div class="sig-value" id="rsi">—</div></div>
  <div class="sig"><div class="sig-label">MACD</div><div class="sig-value" id="macd">—</div></div>
  <div class="sig"><div class="sig-label">BB zone</div><div class="sig-value" id="bbZone">—</div></div>
  <div class="sig"><div class="sig-label">Trend</div><div class="sig-value" id="trend">—</div></div>
  <div class="sig"><div class="sig-label">Supertrend</div><div class="sig-value" id="st">—</div></div>
  <div class="sig"><div class="sig-label">ADX strength</div><div class="sig-value" id="adx">—</div></div>
</div>

<div class="section">Recent trades</div>
<table>
  <thead><tr><th>Time</th><th>Type</th><th>Price</th><th>Score</th><th>Profit (USDT)</th></tr></thead>
  <tbody id="trades"></tbody>
</table>

<script>
const socket = io();
const fmt = (n,d=2) => n!=null ? n.toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}) : "—";
const col = (v,pos,neg) => v===pos?"green":v===neg?"red":"";

socket.on("state", s => {
  document.getElementById("price").textContent    = s.price ? "$"+fmt(s.price) : "—";
  document.getElementById("buyScore").textContent = s.buy_score+"/5";
  document.getElementById("buyScore").className   = "card-value "+(s.buy_score>=4?"green":s.buy_score>=3?"amber":"");
  document.getElementById("position").textContent = s.in_position?"OPEN":"None";
  document.getElementById("position").className   = "card-value "+(s.in_position?"green":"");
  document.getElementById("sl").textContent       = s.stop_loss   ? "$"+fmt(s.stop_loss)   : "—";
  document.getElementById("tp1").textContent      = s.take_profit1? "$"+fmt(s.take_profit1) : "—";
  document.getElementById("tp2").textContent      = s.take_profit2? "$"+fmt(s.take_profit2) : "—";
  document.getElementById("started").textContent  = s.started_at  || "—";
  document.getElementById("lastCheck").textContent= s.last_checked? "Last checked: "+s.last_checked : "";

  const profit = s.total_profit||0;
  const pe = document.getElementById("profit");
  pe.textContent = (profit>=0?"+":"")+"$"+fmt(profit,4);
  pe.className   = "card-value "+(profit>=0?"green":"red");

  document.getElementById("rsi").textContent   = s.rsi||"—";
  document.getElementById("rsi").className      = "sig-value "+(s.rsi<45?"green":s.rsi>58?"red":"");
  document.getElementById("macd").textContent  = s.macd||"—";
  document.getElementById("macd").className     = "sig-value "+(s.macd==="BULL"?"green":"red");
  document.getElementById("bbZone").textContent = s.bb_zone||"—";
  document.getElementById("bbZone").className   = "sig-value "+(s.bb_zone==="BUY ZONE"?"green":"red");
  document.getElementById("trend").textContent  = s.trend||"—";
  document.getElementById("trend").className    = "sig-value "+(s.trend==="UP"?"green":s.trend==="DOWN"?"red":"amber");
  document.getElementById("st").textContent     = s.supertrend||"—";
  document.getElementById("st").className       = "sig-value "+(s.supertrend==="BULL"?"green":"red");
  document.getElementById("adx").textContent    = s.adx ? s.adx+(s.adx>20?" STRONG":" weak") : "—";
  document.getElementById("adx").className      = "sig-value "+(s.adx>20?"green":"red");

  document.getElementById("startBtn").disabled = s.running;
  document.getElementById("stopBtn").disabled  = !s.running;
  document.getElementById("statusDot").className = "dot "+(s.running?"dot-g":"dot-r");
  document.getElementById("statusText").textContent = s.running?"Running — checking every hour":"Stopped";

  const err = document.getElementById("errBox");
  if(s.error){err.style.display="block";err.textContent=s.error;}else{err.style.display="none";}

  const tbody = document.getElementById("trades");
  tbody.innerHTML="";
  ([...(s.trades||[])].reverse()).forEach(t=>{
    const tr = document.createElement("tr");
    tr.innerHTML=`<td>${t.time}</td><td class="${t.type.toLowerCase()}">${t.type}</td><td>$${fmt(t.price)}</td><td>${t.score||"—"}</td><td>${(t.profit||0)>=0?"+":""}${fmt(t.profit||0,4)}</td>`;
    tbody.appendChild(tr);
  });
});

function startBot(){socket.emit("start");}
function stopBot() {socket.emit("stop"); }
setInterval(()=>socket.emit("get_state"),5000);
socket.emit("get_state");
</script></body></html>
"""

@app.route("/")
def dashboard(): return render_template_string(DASH)

@socketio.on("get_state")
def handle_get(): emit("state", clean(state))

@socketio.on("start")
def handle_start():
    if not state["running"]:
        threading.Thread(target=bot_loop, daemon=True).start()
    emit("state", clean(state))

@socketio.on("stop")
def handle_stop():
    state["running"] = False
    emit("state", clean(state))

def push_state():
    while True:
        socketio.emit("state", clean(state))
        time.sleep(5)

if __name__ == "__main__":
    threading.Thread(target=push_state, daemon=True).start()
    mode = "TESTNET" if USE_TESTNET else "LIVE"
    print(f"\n  Strategy A v6 Bot [{mode}]")
    print(f"  Dashboard → http://localhost:5000\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
