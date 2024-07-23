import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from agent_template import TradingAgent  # Assuming you have a base TradingAgent class

class SARIMAXAgent(TradingAgent):
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), initial_cash=100000):
        super().__init__(initial_cash)
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None

    def train_model(self, data):
        self.model = SARIMAX(data['Close'], order=self.order, seasonal_order=self.seasonal_order)
        self.results = self.model.fit(disp=False)
    
    def generate_signals(self, data):
        if self.model is None:
            self.train_model(data)
        prediction = self.results.get_forecast(steps=1).predicted_mean.iloc[-1]
        last_close = data['Close'].iloc[-1]
        if prediction > last_close:
            return 1  # Buy signal
        elif prediction < last_close:
            return -1  # Sell signal
        return 0  # Hold

def fetch_historical_data(symbol, interval, limit=1000):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def prepare_data(data):
    data['Returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    return data

def backtest(agent, data):
    agent.train_model(data)
    for timestamp, row in data.iterrows():
        agent.trade(data.loc[:timestamp])
    return agent.get_portfolio_value()

if __name__ == "__main__":
    # Fetch data for different intervals
    df_1m = fetch_historical_data('BTCUSDT', '1m')
    df_1h = fetch_historical_data('BTCUSDT', '1h')
    df_4h = fetch_historical_data('BTCUSDT', '4h')
    df_1d = fetch_historical_data('BTCUSDT', '1d')

    df_1m = prepare_data(df_1m)
    df_1h = prepare_data(df_1h)
    df_4h = prepare_data(df_4h)
    df_1d = prepare_data(df_1d)

    # Initialize agents
    agent_1m = SARIMAXAgent()
    agent_1h = SARIMAXAgent()
    agent_4h = SARIMAXAgent()
    agent_1d = SARIMAXAgent()

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
        return pd.concat([data, new_data])

    def update_agents():
        global df_1m, df_1h, df_4h, df_1d
        df_1m = update_data(df_1m, '1m')
        df_1h = update_data(df_1h, '1h')
        df_4h = update_data(df_4h, '4h')
        df_1d = update_data(df_1d, '1d')

        agent_1m.trade(df_1m)
        agent_1h.trade(df_1h)
        agent_4h.trade(df_4h)
        agent_1d.trade(df_1d)

        print(f"1m Interval Portfolio Value: {agent_1m.get_portfolio_value()}")
        print(f"1h Interval Portfolio Value: {agent_1h.get_portfolio_value()}")
        print(f"4h Interval Portfolio Value: {agent_4h.get_portfolio_value()}")
        print(f"1d Interval Portfolio Value: {agent_1d.get_portfolio_value()}")

    while True:
        update_agents()
        time.sleep(60)  # Adjust the sleep time according to the interval
