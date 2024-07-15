# -*- coding: utf-8 -*-
"""
@author: Ugur Okumus 
baseline code taken from Harnick Khera (Github.com/Hephyrius)

"""

import numpy as np
from numpy import *
import pandas as pd
import os
from datetime import datetime

from binance.client import Client
from binance.enums import *
import yfinance as yf

import CoreFunctions as cf

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from joblib import dump, load

#You don't need to enter your key/secret in order to get data from the exchange, its only needed for trades in the TradingBot.py class.
api_key = '0'
api_secret = '0'
currency = 'BTCUSDT'
client = Client(api_key, api_secret)

interval_list = [
    ('1m', Client.KLINE_INTERVAL_1MINUTE),
    ('3m', Client.KLINE_INTERVAL_3MINUTE),
    ('5m', Client.KLINE_INTERVAL_5MINUTE),
    ('15m', Client.KLINE_INTERVAL_15MINUTE),
    ('30m', Client.KLINE_INTERVAL_30MINUTE),
    ('1h', Client.KLINE_INTERVAL_1HOUR),
    ('2h', Client.KLINE_INTERVAL_2HOUR),
    ('4h', Client.KLINE_INTERVAL_4HOUR),
    ('6h', Client.KLINE_INTERVAL_6HOUR),
    ('8h', Client.KLINE_INTERVAL_8HOUR),
    ('12h', Client.KLINE_INTERVAL_12HOUR),
    ('1d', Client.KLINE_INTERVAL_1DAY),
    ('3d', Client.KLINE_INTERVAL_3DAY),
    ('1w', Client.KLINE_INTERVAL_1WEEK),
    ('1M', Client.KLINE_INTERVAL_1MONTH)
]

interval_mapping = {item[0]: item[1] for item in interval_list}

def convert_date_format(date_str):
    try:
        return datetime.strptime(date_str, "%d %b, %Y").strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Incorrect date format for {date_str}. Expected 'DD MMM, YYYY'.")

def get_historical_data(interval, start_date, end_date):

    file_name = f"btc_usdt_{interval_mapping[interval]}.csv"
    file_path = os.path.join("Data", file_name)
    target_file_path = file_path.replace(".csv", "_targets.csv")

    if os.path.exists(file_path) and os.path.exists(target_file_path):
        x = pd.read_csv(file_path)
        y = pd.read_csv(target_file_path)['target'].tolist()
    else:
        # Fetch historical data
        candles = client.get_historical_klines(currency, interval_mapping[interval], start_date, end_date)
        
        # Convert the raw data from the exchange into a friendlier form with some basic feature creation
        x = cf.FeatureCreation(candles)

        #cf.add_symbol_close_to_dataframe(client,'PAXGUSDT', x, interval_mapping[interval], start_date, end_date)
        # Create our targets
        offset = 1
        y = cf.CreateTargets(x, offset)
        x = x.iloc[:-offset]


        # Save the feature dataframe and target list to CSV
        x.to_csv(file_path, index=False)
        pd.DataFrame(y, columns=['target']).to_csv(target_file_path, index=False)

    return x, np.array(y)


# Function to fetch latest kline data and convert to feature DataFrame
def get_latest_data(interval, limit=1):
    """
    Fetch recent kline data and convert it to a feature DataFrame.
    
    Parameters:
    - symbol (str): The trading pair symbol, e.g., 'BTCUSDT'.
    - interval (str): The time interval for each candlestick/kline, e.g., '1h'.
    - limit (int): The maximum number of data points to retrieve (default: 100).
    """
    params = {
        'symbol': currency,
        'interval': interval_mapping[interval],
        'limit': limit
    }
    
    candles = client.get_klines(**params)
    
    # Convert the raw data into a friendlier form with some basic feature creation
    feature_data = cf.FeatureCreation(candles)
    
    return feature_data