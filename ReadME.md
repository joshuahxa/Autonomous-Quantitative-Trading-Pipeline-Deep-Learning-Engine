# Autonomous Quant: LSTM Algorithmic Trading System

A professional-grade, multi-asset algorithmic trading architecture powered by a PyTorch LSTM neural network. This system is designed for autonomous execution, active risk management, and capital preservation across distinct macroeconomic regimes.

## 🧠 System Architecture

*   **Neural Core (PyTorch):** A 64-layer Long Short-Term Memory (LSTM) network trained to identify multi-day momentum trends.
*   **Data Pipeline (yfinance):** Utilizes fully split-adjusted, dividend-adjusted historical data to eliminate lookahead bias and artificial stock-split anomalies.
*   **Multi-Regime Training:** The model is forged on 11 years (2010–2021) of continuous data, forcing the neural network to learn from zero-interest bull markets, the 2018 rate-hike taper, and the 2020 flash crash.

## 🛡️ Quantitative Risk Management

Unlike standard momentum bots, this system features institutional-grade guardrails:
*   **Dynamic Macro Filter (VIX):** The algorithm actively polls the S&P 500 Volatility Index. If the VIX spikes above 25 (market panic), the AI dynamically raises its required trade confidence from 60% to 80%.
*   **Hard Stop-Loss Limit:** A mechanical 5% stop-loss prevents the model from "holding the bag" during severe macro downtrends.
*   **Algorithmic Cooldowns:** To prevent "whipsawing" (buying into false dead-cat bounces), any asset that triggers a stop-loss is locked out of trading for a strict 10-day cooldown period.

## ⚙️ Live Execution & Deployment

*   **Broker Integration:** Executes fractional, paper-traded market orders via the Alpaca Trading API.
*   **Real-Time Telemetry:** Integrated with the Telegram API for instant push notifications on model confidence, trade executions, and stop-loss triggers.

## 🚀 Files
*   `train.py`: The data engineering and PyTorch training engine.
*   `backtest.py`: The isolated historical stress-tester (verified against the 2022 bear market).
*   `algo_brain.py`: The live execution and telemetry script.
