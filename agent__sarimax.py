import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class SARIMAXAgent:
    def __init__(self, order, seasonal_order):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None

    def train_model(self, data, exog_data):
        exog_data = exog_data.reindex(data.index).ffill().bfill()
        if data.isnull().values.any() or exog_data.isnull().values.any():
            raise ValueError("Data or exogenous variables contain NaNs")
        if np.isinf(data).values.any() or np.isinf(exog_data).values.any():
            raise ValueError("Data or exogenous variables contain infinite values")

        self.model = SARIMAX(data['Close'], trend='t', order=self.order, seasonal_order=self.seasonal_order, exog=exog_data)
        self.results = self.model.fit(disp=False)

    def generate_signals(self, data, exog_data):
        exog_data = exog_data.reindex(data.index).ffill().bfill()
        if exog_data.isnull().values.any():
            raise ValueError("Exogenous variables contain NaNs after reindexing")
        if np.isinf(exog_data).values.any():
            raise ValueError("Exogenous variables contain infinite values after reindexing")
        
        prediction = self.results.get_forecast(steps=1, exog=exog_data.iloc[-1:]).predicted_mean.iloc[-1]
        return prediction

    def trade(self, data, exog_data):
        signal = self.generate_signals(data, exog_data)
        print(f"Generated Signal: {signal}")
        return signal

def backtest(agent, data, exog_data):
    actual = []
    predicted = []
    portfolio_value = 100000  # Starting portfolio value in USD
    position = 0  # Current position in BTC
    buy_threshold = 0.01  # Threshold to trigger buy signal
    sell_threshold = -0.01  # Threshold to trigger sell signal

    for timestamp in data.index:
        agent.train_model(data.loc[:timestamp], exog_data.loc[:timestamp])
        signal = agent.trade(data.loc[:timestamp], exog_data)
        actual_price = data.loc[timestamp, 'Close']
        actual.append(actual_price)
        predicted.append(signal)
        
        if signal - actual_price > buy_threshold:
            position += 1  # Buy 1 BTC
            portfolio_value -= actual_price  # Subtract the price of 1 BTC from portfolio value
        elif actual_price - signal > sell_threshold and position > 0:
            position -= 1  # Sell 1 BTC
            portfolio_value += actual_price  # Add the price of 1 BTC to portfolio value

        print(f"Timestamp: {timestamp}, Signal: {signal}, Actual Price: {actual_price}, Position: {position}, Portfolio Value: {portfolio_value}")

    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    
    print(f'R-squared: {r2}')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    
    return portfolio_value

# Generate synthetic exogenous data
def generate_synthetic_exog_data(start_date, end_date, freq):
    date_rng = pd.date_range(start=start_date, end=end_date, freq=freq)
    exog_data = pd.DataFrame(date_rng, columns=['timestamp'])
    exog_data['synthetic_feature'] = np.random.randn(len(date_rng))
    exog_data.set_index('timestamp', inplace=True)
    return exog_data

# Generate synthetic market data
def generate_synthetic_market_data(start_date, end_date, freq):
    date_rng = pd.date_range(start=start_date, end=end_date, freq=freq)
    market_data = pd.DataFrame(date_rng, columns=['timestamp'])
    market_data['Close'] = np.cumsum(np.random.randn(len(date_rng)))
    market_data.set_index('timestamp', inplace=True)
    return market_data

# Example usage
if __name__ == "__main__":
    # Replace with your start and end date for synthetic data
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    freq = 'H'  # Hourly frequency

    df_1h = generate_synthetic_market_data(start_date, end_date, freq)
    exog_data = generate_synthetic_exog_data(start_date, end_date, freq)

    agent_1h = SARIMAXAgent(order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
    final_portfolio_value = backtest(agent_1h, df_1h, exog_data)
    print(f"Final Portfolio Value: {final_portfolio_value}")
