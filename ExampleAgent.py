from TradingAgent import TradingAgent
import numpy as np
from catboost import Pool, CatBoostRegressor

class ExampleAgent(TradingAgent):
    def __init__(self, name):
        """
        Initializes the ExampleAgent instance with a given name and using CatBoostRegressor as the model.
        
        Parameters:
            name (str): Agent name
            model (object): Model to be used for trading
        """        
        super().__init__(name, CatBoostRegressor())

    def generate_signals(self, data):
        """
        Generates trading signals based on the current and predicted prices.

        Parameters:
            data (DataFrame): The input data containing the close prices or returns.

        Returns:
            int: The signal indicating whether to Hold, Buy, or Sell (0: Hold, 1: Buy, 2: Sell).
        """
        # Change here
        current_price = data['close'].iloc[-1]
        pred_price = self.model.predict(np.array([current_price]).reshape(1, -1))[0]
        if pred_price > current_price:
            return 1
        elif pred_price < current_price:
            return 2
        else:
            return 0
    
    def train_model(self, data):
        """
        Trains the model using the provided data.

        Parameters:
            data (DataFrame): The input data containing the close prices.
        """
        # Change here
        data = data['close']
        train_data = data
        train_label = train_data.shift(-1).dropna()
        train_data = train_data[:-1]
        train_pool = Pool(train_data, train_label)
        self.model.fit(train_pool, silent=True)

    def trade(self, data):
        super().trade(data)
    
    def get_portfolio_value(self, current_price):
        return super().get_portfolio_value(current_price)