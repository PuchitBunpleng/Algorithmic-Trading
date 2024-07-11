import requests
import pandas as pd
import numpy as np
import time

from DummyAgent import DummyAgent

def fetch_historical_data(symbol, interval, limit=1000):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# Fetch data for different intervals
df_1m = fetch_historical_data('BTCUSDT', '1m')
df_1h = fetch_historical_data('BTCUSDT', '1h')
df_4h = fetch_historical_data('BTCUSDT', '4h')
df_1d = fetch_historical_data('BTCUSDT', '1d')

# Prepare the training data
def prepare_data(data):
    data['returns'] = data['close'].pct_change()
    data.dropna(inplace=True)
    return data

df_1m = prepare_data(df_1m)
df_1h = prepare_data(df_1h)
df_4h = prepare_data(df_4h)
df_1d = prepare_data(df_1d)

# Backtesting function
def backtest(agent, data):
    agent.train_model(data)
    for timestamp, row in data.iterrows():
        agent.trade(data.loc[:timestamp])
    return agent.get_portfolio_value(row['close'])

# Initialize agents
agent_1m = DummyAgent('Dummy Agent 1m')
agent_1h = DummyAgent('Dummy Agent 1h')
agent_4h = DummyAgent('Dummy Agent 4h')
agent_1d = DummyAgent('Dummy Agent 1d')

# Backtest each agent
portfolio_value_1m = backtest(agent_1m, df_1m)
portfolio_value_1h = backtest(agent_1h, df_1h)
portfolio_value_4h = backtest(agent_4h, df_4h)
portfolio_value_1d = backtest(agent_1d, df_1d)
print(f"Portfolio Value for 1m Interval: {portfolio_value_1m}")
print(f"Portfolio Value for 1h Interval: {portfolio_value_1h}")
print(f"Portfolio Value for 4h Interval: {portfolio_value_4h}")
print(f"Portfolio Value for 1d Interval: {portfolio_value_1d}")