# Autonomous Quant Trader: LSTM Neural Network

An end-to-end algorithmic trading pipeline that uses a Long Short-Term Memory (LSTM) neural network to analyze market kinematics and autonomously execute trades. 

## Architecture
This project is divided into two primary engines:
1. **The Training Engine (`ai_trainer.py`):** Ingests historical market data via the Alpaca API, calculates technical indicators (Volume, SMA, RSI), and normalizes the features. It trains a PyTorch LSTM model to calculate directional market probability and saves the optimized network weights.
2. **The Execution Engine (`algo_brain.py`):** Designed to run as a headless background task. It loads the serialized `.pth` brain, pulls live market data, constructs a multivariable tensor, runs a forward inference pass, and executes trades if mathematical confidence thresholds are met.

## Tech Stack
* **Language:** Python
* **Machine Learning:** PyTorch, Scikit-learn (MinMaxScaler)
* **Data Processing:** Pandas, NumPy
* **Brokerage Integration:** Alpaca Trading API

## Mathematical Foundation
The model treats financial markets as sequential kinematic systems. Instead of predicting exact scalar prices (regression), the neural network is constrained to a binary classification problem using a Sigmoid activation function, outputting a strict probability matrix bounded between $0.0$ and $1.0$.