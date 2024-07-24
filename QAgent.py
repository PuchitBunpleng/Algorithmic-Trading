import numpy as np
import pandas as pd
from TradingAgent import TradingAgent

class QLearningAgent(TradingAgent):
    def __init__(self, name, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.3, num_episodes=500):
        """
        Initializes the QLearningAgent instance with a given name and Q-learning parameters.
        
        Parameters:
            name (str): Agent name
            learning_rate (float): Learning rate for Q-learning
            discount_factor (float): Discount factor for Q-learning
            exploration_prob (float): Probability of exploration in epsilon-greedy policy
            num_episodes (int): Number of episodes for training
        """
        super().__init__(name)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.num_episodes = num_episodes
        self.actions = ['buy', 'sell', 'hold']
        self.Q = None

    def generate_signals(self, data, num_bins=10):
        """
        Generates trading signals based on the Q-table.

        Parameters:
        data (DataFrame): The input data containing the prices or returns.
        num_bins (int): The number of bins to use for discretizing the prices.

        Returns:
        int: The signal indicating whether to Hold, Buy, or Sell (0: Hold, 1: Buy, 2: Sell).
        """
        prices = data['close'].values
        binned_prices = self.bin_prices(prices, num_bins)
        state = binned_prices[-1]
        action = self.actions[np.argmax(self.Q[state])]
        print(action)
        if action == 'buy':
            return 1
        elif action == 'sell':
            return 2
        else:
            return 0


    def train_model(self, data, num_bins=10):
        """
        Trains the Q-learning model using the provided data with frequency-based binning.

        Parameters:
        data (DataFrame): The input data containing the prices or returns.
        num_bins (int): The number of bins to use for discretizing the prices.
        """
        prices = data['close'].values
        binned_prices = self.bin_prices(prices, num_bins)
        states = range(len(binned_prices))
        self.Q = np.zeros((num_bins, len(self.actions)))  # Use num_bins instead of len(states)


        for episode in range(self.num_episodes):
            state = binned_prices[0]
            shares_held = 0
            cash = self.cash
            done = False

            while not done:
                if np.random.uniform(0, 1) < self.exploration_prob:
                    action = np.random.choice(self.actions)
                else:
                    action = self.actions[np.argmax(self.Q[state])]

                next_state, reward, shares_held, cash = self.simulate_trading_environment(state, action, shares_held, cash, binned_prices)

                best_next_action = np.argmax(self.Q[next_state])
                self.Q[state, self.actions.index(action)] = self.Q[state, self.actions.index(action)] + self.learning_rate * (reward + self.discount_factor * self.Q[next_state, best_next_action] - self.Q[state, self.actions.index(action)])

                state = next_state
                if state == len(binned_prices) - 1:
                    done = True


    def prepare_data(self, data):
        """
        Prepare the training data.

        Parameters:
            data (pandas.DataFrame): The input data containing the prices or returns.

        Returns:
            tuple: A tuple containing the training data and the corresponding labels.
                - train_data (pandas.DataFrame): The training data.
                - train_label (pandas.Series): The labels data.
        """
        return data, data['close']

    def extract_feature(self, data):
        """
        Extracts features from the data.

        Parameters:
            data (pandas.DataFrame): The input data containing the prices or returns.

        Returns:
            list: A list of features extracted from the data.
        """
        return data['close'].values.tolist()

    def trade(self, data):
        """
        Executes a trade based on the given data.

        Parameters:
            data (DataFrame): The input data containing the close prices.
        """
        timestamp = data.index[-1]
        if len(data) < self.num_episodes:
            print(f"{timestamp}: {self.name} - Collecting data...")
            return
        if len(data) % self.num_episodes == 0:
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

    def simulate_trading_environment(self, state, action, shares_held, cash, binned_prices):
        """
        Simulates the trading environment and returns the next state, reward, shares held, and cash.

        Parameters:
        state (int): The current state.
        action (str): The action to take ('buy', 'sell', 'hold').
        shares_held (int): The number of shares held.
        cash (float): The amount of cash available.
        binned_prices (array): The array of binned prices.

        Returns:
        tuple: The next state, reward, shares held, and cash.
        """
        next_state = state + 1 if state < len(binned_prices) - 1 else state
        reward = 0
        price = binned_prices[state]
        next_price = binned_prices[next_state]

        if action == 'buy' and cash >= price:
            shares_bought = cash // price
            cash -= shares_bought * price
            shares_held += shares_bought
            reward = 0  # No immediate reward for buying
        elif action == 'sell' and shares_held > 0:
            cash += shares_held * price
            reward = shares_held * (next_price - price)  # Profit from selling
            shares_held = 0
        elif action == 'hold':
            reward = 0  # No immediate reward for holding

        return next_state, reward, shares_held, cash



    def bin_prices(self, prices, num_bins):
        """
        Bin the prices into discrete bins using frequency-based binning.

        Parameters:
        prices (array-like): The array of prices.
        num_bins (int): The number of bins to use.

        Returns:
            array: The binned prices.
        """
        bins = pd.qcut(prices, num_bins, labels=False)
        return bins


