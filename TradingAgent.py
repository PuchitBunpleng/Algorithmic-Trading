from abc import abstractmethod

class TradingAgent:
    def __init__(self, name):
        """
        Initializes the ExampleAgent instance with a given name.
        
        Parameters:
            name (str): Agent name
        """     
        self.name = name
        self.position = 0  # 0: No position, 1: Long, -1: Short
        self.cash = 100000  # Starting cash in USD
        self.holdings = 0

    @abstractmethod
    def generate_signals(self, data):
        """
        Generates trading signals based on the current and predicted prices.

        Parameters:
            data (DataFrame): The input data containing the prices or returns.

        Returns:
            int: The signal indicating whether to Hold, Buy, or Sell (0: Hold, 1: Buy, 2: Sell).
        """
        pass

    @abstractmethod
    def train_model(self, data):
        """
        Trains the model using the provided data.

        Parameters:
            data (DataFrame): The input data containing the prices or returns.
        """
        pass

    @abstractmethod
    def prepare_data(self, data):
        """
        Prepare the training data

        Parameters:
            data (pandas.DataFrame): The input data containing the prices or returns.

        Returns:
            tuple: A tuple containing the training data and the corresponding labels.
                - train_data (pandas.DataFrame): The training data.
                - train_label (pandas.Series): The labels data.
        """
        pass

    @abstractmethod
    def trade(self, data):
        """
        Executes a trade based on the given data.

        Args:
            data (DataFrame): The input data containing the prices or returns.
        """
        pass

    @abstractmethod
    def extract_feature(self, data):
        pass

    def get_portfolio_value(self, current_price):
        return self.cash + (self.holdings * current_price)
    
    def reset(self):
        self.position = 0
        self.cash = 100000
        self.holdings = 0
