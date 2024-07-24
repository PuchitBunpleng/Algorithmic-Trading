import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from DummyAgent import TradingAgent  # Assuming you have a base TradingAgent class

class SARIMAXAgent(TradingAgent):
    def __init__(self, name, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), exog_vars=None, buy_threshold=0, sell_threshold=0):
        super().__init__(name)
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_vars = exog_vars
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.model = None
        self.retrain_period = 500

    def train_model(self, data):
        print(self.order)
        if self.exog_vars:
            exog = data[self.exog_vars]
            self.model = SARIMAX(data['close'], exog=exog, order=self.order, seasonal_order=self.seasonal_order)
        else:
            self.model = SARIMAX(data['close'], order=self.order, seasonal_order=self.seasonal_order)
        self.results = self.model.fit(disp=False)
    
    def generate_signals(self, data):
        if self.exog_vars:
            exog = data[self.exog_vars].iloc[-1:]
            prediction = self.results.get_forecast(steps=1, exog=exog).predicted_mean.iloc[-1]
        else:
            prediction = self.results.get_forecast(steps=1).predicted_mean.iloc[-1]
        current = data['close'].iloc[-1]
        if prediction - current > self.buy_threshold:
            return 1  # Buy signal
        elif prediction - current < -self.sell_threshold:
            return 2  # Sell signal
        return 0  # Hold

    def trade(self, data):
        timestamp = data.index[-1]
        if len(data) < self.retrain_period:
            print(f"{timestamp}: {self.name} - Collecting data...")
            return
        if len(data) % self.retrain_period == 0:
            data = data[-self.retrain_period:]  
            self.train_model(data)
            print(f"{timestamp}: {self.name} - Updated model")
        signal = self.generate_signals(data)
        price = data['close'].iloc[-1]
        if signal == 1 and self.position != 1 and self.cash > 0:
            self.holdings = self.cash / price
            self.cash = 0
            self.position = 1
            print(f"{timestamp}: {self.name} - Buy at {price}")
        elif signal == 2 and self.position != -1 and self.holdings > 0:
            self.cash = self.holdings * price
            self.holdings = 0
            self.position = -1
            print(f"{timestamp}: {self.name} - Sell at {price}")
        else:
            print(f"{timestamp}: {self.name} - Hold")

    def evaluate_model(self, data):
        if self.exog_vars:
            exog = data[self.exog_vars]
            predictions = self.results.predict(start=0, end=len(data)-1, exog=exog)
        else:
            predictions = self.results.predict(start=0, end=len(data)-1)
        true_values = data['close']
        r_squared = r2_score(true_values, predictions)
        mae = mean_absolute_error(true_values, predictions)
        mse = mean_squared_error(true_values, predictions)
        print(f"R-squared: {r_squared}")
        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        return r_squared, mae, mse

    def get_portfolio_value(self, current_price):
        return super().get_portfolio_value(current_price)

# Example usage:
# agent = SARIMAXAgent(name="SARIMAX_Trader", exog_vars=['exog1', 'exog2'])
# agent.train_model(data)
# agent.evaluate_model(data)
# while trading:
#     agent.trade(new_data)
