from TradingAgent import TradingAgent
import numpy as np

class DummyAgent(TradingAgent):
    def __init__(self, name):
        super().__init__(name)

    def generate_signals(self, data):
        return np.random.choice([0, 1, 2])
    
    def train_model(self, data):
        pass

    def trade(self, data):
        timestamp = data.index[-1]
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
        self.history.append(self.get_portfolio_value(price))
    
    def get_portfolio_value(self, current_price):
        return super().get_portfolio_value(current_price)