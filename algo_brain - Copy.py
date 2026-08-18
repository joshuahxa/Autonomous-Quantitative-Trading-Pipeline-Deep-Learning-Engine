import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import time
import sys

# 1. Automated Background Logging
log_path = r"C:\Users\joshu\Downloads\algo_log.txt"
sys.stdout = open(log_path, "a")
sys.stderr = sys.stdout

print(f"\n=== AI EXECUTION RUN: {datetime.now()} ===")

# 2. API Credentials (PAPER KEYS)
API_KEY = "YOUR_PAPER_API_KEY"
SECRET_KEY = "YOUR_PAPER_SECRET_KEY"

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# 3. Define the AI Architecture (Must match the trainer exactly)
class TradingLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        return self.sigmoid(self.linear(lstm_out[:, -1, :]))

# 4. Load the Trained Neural Network
brain_path = r"C:\Users\joshu\Downloads\trade_brain.pth"
model = TradingLSTM(input_size=5)
model.load_state_dict(torch.load(brain_path, weights_only=True))
model.eval() # Set to evaluation mode (turns off learning)
print("[+] Neural Brain successfully loaded into execution memory.")

def get_engineered_data(symbol):
    """Pulls recent data and calculates the exact features the AI requires."""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=100)
    
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start_time, end=end_time)
    df = data_client.get_stock_bars(req).df
    
    # Calculate Features
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_30'] = df['close'].rolling(window=30).mean()
    
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df.dropna(inplace=True)
    return df

def ai_strategy(symbol):
    print(f"\n=== INFERENCING {symbol} ===")
    
    try:
        df = get_engineered_data(symbol)
    except Exception as e:
        print(f"[-] Data fetch failed: {e}")
        return

    # Isolate features and scale them between 0 and 1
    features = df[['close', 'volume', 'SMA_10', 'SMA_30', 'RSI']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)
    
    # Grab only the last 10 days to create the sequence for the AI
    recent_sequence = scaled_data[-10:]
    
    # Convert to PyTorch Tensor: Shape (Batch=1, Sequence=10, Features=5)
    tensor_input = torch.tensor(recent_sequence, dtype=torch.float32).unsqueeze(0)
    
    # Run the math
    with torch.no_grad():
        ai_confidence = model(tensor_input).item()
        
    print(f"[*] AI Upward Probability: {ai_confidence * 100:.2f}%")
    
    # Check Portfolio
    try:
        position = trading_client.get_open_position(symbol)
        current_qty = float(position.qty)
        print(f"[*] Position: Holding {current_qty} shares.")
    except Exception:
        current_qty = 0
        print("[*] Position: No shares held.")

    # Execute logic based on strict confidence thresholds
    if ai_confidence > 0.60: # AI is over 60% sure it will go up
        if current_qty == 0:
            print("[!] HIGH CONFIDENCE BULLISH. Submitting BUY order...")
            order = MarketOrderRequest(symbol=symbol, qty=1, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
            trading_client.submit_order(order_data=order)
        else:
            print("[*] AI is Bullish, but we are already holding.")
            
    elif ai_confidence < 0.45: # AI thinks there's less than a 45% chance it goes up (Bearish)
        if current_qty > 0:
            print("[!] HIGH CONFIDENCE BEARISH. Submitting SELL order...")
            order = MarketOrderRequest(symbol=symbol, qty=current_qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
            trading_client.submit_order(order_data=order)
        else:
            print("[*] AI is Bearish, but no shares to sell.")
            
    else:
        print("[*] AI is uncertain (Probability in neutral zone). Holding current state.")

if __name__ == "__main__":
    clock = trading_client.get_clock()
    if not clock.is_open:
        print("[-] Notice: Market closed. Orders will queue.")
        
    target_stocks = ["AAPL", "NVDA", "TSLA", "MSFT", "AMD"]
    
    for stock in target_stocks:
        ai_strategy(stock)
        time.sleep(1)
        
    print("\n=== AI SCAN COMPLETE ===")