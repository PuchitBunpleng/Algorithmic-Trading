from abc import abstractmethod
import pandas as pd

class TradingAgent:
    def __init__(self, name, model=None):
        self.name = name
        self.position = 0  # 0: No position, 1: Long, -1: Short
        self.cash = 100000  # Starting cash in USD
        self.holdings = 0
        self.model = model

    @abstractmethod
    def generate_signals(self, data):
        # Return
        # 0: Hold, 1: Buy, 2: Sell
        pass

    @abstractmethod
    def train_model(self, data):
        pass

    def trade(self, data):
        signal = self.generate_signals(data)
        price = data['close'].iloc[-1]
        if signal == 1 and self.position != 1:
            if self.cash > 0:
                self.holdings = self.cash / price
                self.cash = 0
                self.position = 1
                print(f"{pd.Timestamp.now()}: {self.name} - Buy at {price}")
        elif signal == 2 and self.position != -1:
            if self.holdings > 0:
                self.cash = self.holdings * price
                self.holdings = 0
                self.position = -1
                print(f"{pd.Timestamp.now()}: {self.name} - Sell at {price}")
        else:
            print(f"{pd.Timestamp.now()}: {self.name} - Hold")

    def get_portfolio_value(self, current_price):
        return self.cash + (self.holdings * current_price)
