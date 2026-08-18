import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

# 1. API Credentials (PAPER)
API_KEY = "YOUR_PAPER_API_KEY"
SECRET_KEY = "YOUR_PAPER_SECRET_KEY"
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

def get_training_data(symbol, years=3):
    print(f"[*] Downloading data for {symbol}...")
    end_time = datetime.now()
    start_time = end_time - timedelta(days=years*365)
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start_time, end=end_time
    )
    # The response natively converts to a pandas dataframe
    return data_client.get_stock_bars(request_params).df

def add_technical_indicators(df):
    print("[*] Engineering multivariate features (Volume, SMA, RSI)...")
    # Momentum & Baseline Trajectory
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_30'] = df['close'].rolling(window=30).mean()
    
    # RSI Calculation (Measures overbought/oversold conditions)
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Drop initial rows with NaN values created by the 30-day window
    df.dropna(inplace=True)
    return df

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        # Target: 1.0 if tomorrow's close price (index 0) > today's close price, else 0.0
        y = 1.0 if data[i + seq_length, 0] > data[i + seq_length - 1, 0] else 0.0
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class TradingLSTM(nn.Module):
    # Upgraded input_size to 5 to handle our multivariate features
    def __init__(self, input_size=5, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return self.sigmoid(predictions)

if __name__ == "__main__":
    print("=== STARTING ADVANCED MULTIVARIATE TRAINING ===\n")
    
    # 1. Prepare Data
    df = get_training_data("AAPL")
    df = add_technical_indicators(df)
    
    # Isolate the 5 specific features the AI will learn from
    features = df[['close', 'volume', 'SMA_10', 'SMA_30', 'RSI']].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)
    
    # 2. Create Time Sequences
    SEQ_LENGTH = 10
    X, y = create_sequences(scaled_data, SEQ_LENGTH)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    # 3. Initialize Model (Notice input_size=5)
    model = TradingLSTM(input_size=5)
    loss_function = nn.BCELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Train Model
    epochs = 150
    print(f"[*] Training initiated for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        y_pred = model(X_tensor)
        single_loss = loss_function(y_pred, y_tensor)
        single_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3} | Loss: {single_loss.item():.4f}")
            
    print("\n[+] Training Complete!")
    
    # 5. Export Neural Network Weights to Disk
    save_path = r"C:\Users\joshu\Downloads\trade_brain.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[+] AI Neural Weights successfully saved to: {save_path}")