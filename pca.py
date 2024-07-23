import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class data_preparation():
    
    @staticmethod
    def prepared_data():

        def fetch_historical_data(symbol, interval, limit=10000):
            url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
            response = requests.get(url)
            data = response.json()
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            return df

        def calculate_bollinger_bands(data, window=20, num_std_dev=2):
            data['SMA'] = data['close'].rolling(window=window).mean()
            data['STD'] = data['close'].rolling(window=window).std()
            data['Upper Band'] = data['SMA'] + (data['STD'] * num_std_dev)
            data['Lower Band'] = data['SMA'] - (data['STD'] * num_std_dev)
            return data

        def calculate_rsi(data, window=14):
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.rolling(window=window, min_periods=1).mean()
            avg_loss = loss.rolling(window=window, min_periods=1).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
            short_ema = data['close'].ewm(span=short_window, adjust=False).mean()
            long_ema = data['close'].ewm(span=long_window, adjust=False).mean()
            data['MACD'] = short_ema - long_ema
            data['Signal Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
            data['MACD Histogram'] = data['MACD'] - data['Signal Line']
            return data

        def calculate_cci(data, ndays=20):
            data['TP'] = (data['high'] + data['low'] + data['close']) / 3
            data['sma'] = data['TP'].rolling(ndays).mean()
            data['mad'] = data['TP'].rolling(ndays).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
            data['CCI'] = (data['TP'] - data['sma']) / (0.015 * data['mad'])
            return data

        def prepare_data(data):
            data['returns'] = data['close'].pct_change()
            data.dropna(inplace=True)
            return data

        # Fetch historical data
        data = fetch_historical_data('BTCUSDT', '1d')
        data_close = data['close']
        data = prepare_data(data)
        # Calculate Bollinger Bands
        data = calculate_bollinger_bands(data)

        # Calculate %B
        data['%B'] = (data['close'] - data['Lower Band']) / (data['Upper Band'] - data['Lower Band'])

        # Calculate RSI with a 14-period window
        data['RSI'] = calculate_rsi(data)

        # Calculate MACD
        data = calculate_macd(data)

        # Calculate CCI with a 20-period window
        data = calculate_cci(data, ndays=20)

        data['close'] = data_close
        # Normalize all features using MinMaxScaler
        scaler = MinMaxScaler()
        features_to_normalize = ['close', 'SMA', 'Upper Band', 'Lower Band', '%B', 'RSI', 'MACD', 'Signal Line', 'MACD Histogram', 'CCI', 'returns']
        data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])
        
        # Remove rows with NaN values
        data.dropna(inplace=True)

        data = data[features_to_normalize]
        data_close = data['close']
        # Preserve the timestamp index
        timestamps = data.index

        # Apply PCA
        pca = PCA()
        principal_components = pca.fit_transform(data[features_to_normalize])

        # Calculate cumulative explained variance
        cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

        # Determine the number of components that explain at least 95% of the variance
        n_components = np.argmax(cumulative_explained_variance >= 0.95) + 1

        # Apply PCA with the determined number of components
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(data[features_to_normalize])

        # Create a DataFrame with principal components
        pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)], index=timestamps)
        pca_df['returns'] = data["returns"]

        return pca_df, data 

    
pca_df, data = data_preparation.prepared_data()
print(pca_df.head())
print(data['close'])
