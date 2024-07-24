import requests
import pandas as pd
import time

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

# Initialize agents (Change here)
from another_SARIMAX import SARIMAXAgent

agent_1m = SARIMAXAgent('CatBoost Agent 1m', buy_threshold=0.001, sell_threshold=0.0015)
agent_1h = SARIMAXAgent('CatBoost Agent 1h', buy_threshold=0.01, sell_threshold=0.02)
agent_4h = SARIMAXAgent('CatBoost Agent 4h', buy_threshold=0.012, sell_threshold=0.03)
agent_1d = SARIMAXAgent('CatBoost Agent 1d', buy_threshold=0.05, sell_threshold=0.06)

# Backtest each agent
portfolio_value_1m = backtest(agent_1m, df_1m)
portfolio_value_1h = backtest(agent_1h, df_1h)
portfolio_value_4h = backtest(agent_4h, df_4h)
portfolio_value_1d = backtest(agent_1d, df_1d)
print(f"Portfolio Value for 1m Interval: {portfolio_value_1m}")
print(f"Portfolio Value for 1h Interval: {portfolio_value_1h}")
print(f"Portfolio Value for 4h Interval: {portfolio_value_4h}")
print(f"Portfolio Value for 1d Interval: {portfolio_value_1d}")

# Function to update agents in real-time
def update_data(data, interval):
    new_data = fetch_historical_data('BTCUSDT', interval, limit=1)
    if new_data.index[-1] == data.index[-1]:
        return data
    return pd.concat([data, new_data])

def update_agents():
    global df_1m, df_1h, df_4h, df_1d
    df_1m = update_data(df_1m, '1m')
    df_1h = update_data(df_1h, '1h')
    df_4h = update_data(df_4h, '4h')
    df_1d = update_data(df_1d, '1d')

    df_1m = prepare_data(df_1m)
    df_1h = prepare_data(df_1h)
    df_4h = prepare_data(df_4h)
    df_1d = prepare_data(df_1d)

    agent_1m.trade(df_1m)
    agent_1h.trade(df_1h)
    agent_4h.trade(df_4h)
    agent_1d.trade(df_1d)

    print(f"1m Interval Portfolio Value: {agent_1m.get_portfolio_value(df_1m['close'].iloc[-1])}")
    print(f"1h Interval Portfolio Value: {agent_1h.get_portfolio_value(df_1h['close'].iloc[-1])}")
    print(f"4h Interval Portfolio Value: {agent_4h.get_portfolio_value(df_4h['close'].iloc[-1])}")
    print(f"1d Interval Portfolio Value: {agent_1d.get_portfolio_value(df_1d['close'].iloc[-1])}")

# Reset agents
agent_1m.reset()
agent_1h.reset()
agent_4h.reset()
agent_1d.reset()

# Start trading in real-time
print()
print("*-*-*-*-*-*-*-*-*-*Start trading in real-time*-*-*-*-*-*-*-*-*-*")
while True:
    update_agents()
    time.sleep(60) # Adjust the sleep time according to the interval