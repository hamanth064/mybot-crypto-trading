# MyBot — Automated Crypto Trading Bot

A fully automated BTC/USDT trading bot built 
from scratch with 60% win rate backtested 
over 15 years of data (2011-2026).

## Performance
- Win rate: 60.29%
- Profit factor: 1.555
- Max drawdown: 10.37%
- Total trades: 136 (15 years)
- Backtested: Jan 2011 — Mar 2026

## Tech Stack
- Python 3.11
- Flask + Flask-SocketIO (live dashboard)
- Binance API (python-binance)
- Pandas + TA library
- TradingView Pine Script

## Strategy — 6 Signal Scoring System
Trades only when 3 out of 6 signals agree:
1. RSI (momentum)
2. MACD crossover (trend direction)
3. Bollinger Bands (price zone)
4. EMA 21/55/200 (trend filter)
5. Supertrend (trend confirmation)
6. ADX (trend strength)

## Features
- Live web dashboard at localhost:5000
- Auto buy/sell with stop loss
- Partial take profit (50% at TP1, rest at TP2)
- Trailing stop loss
- Runs on Binance testnet (paper trading)
- Daily timeframe — checks every hour

## How to Run
pip install python-binance flask 
flask-socketio pandas ta

python mybot.py

Open browser: http://localhost:5000
