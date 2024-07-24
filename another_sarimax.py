import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAXAgent:
    def __init__(self, order, seasonal_order):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.results = None

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
    portfolio_value = []
    for timestamp in data.index:
        agent.train_model(data.loc[:timestamp], exog_data.loc[:timestamp])
        signal = agent.trade(data.loc[:timestamp], exog_data)
        portfolio_value.append(signal)
    return portfolio_value

# Example usage
if __name__ == "__main__":
    # Replace '/path/to/your/data.csv' and '/path/to/your/exog_data.csv' with the actual file paths
    df_1h = pd.read_csv('/Users/proudpcy/Documents/GitHub/Algorithmic-Trading/your_data.csv', parse_dates=['timestamp'], index_col='timestamp')
    exog_data = pd.read_csv('/Users/proudpcy/Documents/GitHub/Algorithmic-Trading/your_exog_data.csv', parse_dates=['timestamp'], index_col='timestamp')

    agent_1h = SARIMAXAgent(order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
    portfolio_value_1h = backtest(agent_1h, df_1h, exog_data)
    print(portfolio_value_1h)

