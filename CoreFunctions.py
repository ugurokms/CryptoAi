"""

@author: Ugur Okumus 
baseline code taken from Harnick Khera (Github.com/Hephyrius)

Use this class as your pipeline, use it for all of your data manipulation/feature creation functionality.

Functions here are used across the bot and training classes!

"""
from binance.client import Client
import pandas as pd 
import numpy as np
import ta


#Get the balance of a specified coin
def getCoinBalance(client, currency):
    balance = float(client.get_asset_balance(asset=currency)['free'])
    return balance

#Market buy
def executeBuy(client, market, qtyBuy):
    
    order = client.order_market_buy(symbol=market,quantity=qtyBuy)

#Market sell
def executeSell(client, market, qtySell):

    order = client.order_market_sell(symbol=market, quantity=qtySell)

def create_dataframe(klines):
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(klines, columns=columns)
    
    # Convert timestamps from milliseconds to seconds
    df['timestamp'] = df['timestamp'].astype(float) / 1000
    df['close_time'] = df['close_time'].astype(float) / 1000

    # Convert data types
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
    return df

def FeatureCreation(klines):
    # Convert raw data to a DataFrame
    convertedData = create_dataframe(klines)
    
    # Add technical indicators directly to the convertedData DataFrame
    convertedData['rsi'] = ta.momentum.rsi(convertedData['close'], window=14)
    convertedData['macd'] = ta.trend.macd(convertedData['close'])
    convertedData['macd_signal'] = ta.trend.macd_signal(convertedData['close'])
    convertedData['macd_diff'] = ta.trend.macd_diff(convertedData['close'])
    convertedData['bb_high'] = ta.volatility.bollinger_hband(convertedData['close'])
    convertedData['bb_low'] = ta.volatility.bollinger_lband(convertedData['close'])
    convertedData['bb_mid'] = ta.volatility.bollinger_mavg(convertedData['close'])
    convertedData['bb_high_indicator'] = ta.volatility.bollinger_hband_indicator(convertedData['close'])
    convertedData['bb_low_indicator'] = ta.volatility.bollinger_lband_indicator(convertedData['close'])
    
    return convertedData


#Create targets for our machine learning model. This is done by predicting if the closing price of the next candle will 
#be higher or lower than the current one.
def CreateTargets(data, offset):
    y = []
    
    for i in range(0, len(data)-offset):
        current = float(data[i][3])
        comparison = float(data[i+offset][3])
        
        if current<comparison:
            y.append(1)

        elif current>=comparison:
            y.append(0)
            
    return y

#FEATURE EXAMPLES
#Calculate the change in the values of a column
def GetChangeData(x):

    cols = x.columns
    
    for i in cols:
        j = "c_" + i
        
        try:
            dif = x[i].diff()
            x[j] = dif
        except Exception as e:
            print(e)
            
#FEATURE EXAMPLES  
#Calculate the percentage change between this bar and the previoud x bars
def ChangeTime(x, step):
    
    out = []
    
    for i in range(len(x)):
        try:
            a = x[i]
            b = x[i-step]
            
            change = (1 - b/a) 
            out.append(change)
        except Exception as e:
            out.append(0)
    
    return out

#FEATURE EXAMPLES
#Automate the creation of percentage changes for 48 candles.  
def StepData(x, data):
    
    for i in range(1,48):
        
        data[str(i)+"StepDifference"] = ChangeTime(x, i)


#FEATURE EXAMPLES
#Features that take into acount the relations between the candle values  
def candleRatios(data):
    data['v_c'] = data['v'] / data['c']
    data['h_c'] = data['h'] / data['c']
    data['o_c'] = data['o'] / data['c']
    data['l_c'] = data['l'] / data['c']
    
    data['h_l'] = data['h'] / data['l']
    data['v_l'] = data['v'] / data['l']
    data['o_l'] = data['o'] / data['l']
    
    data['o_h'] = data['o'] / data['h']
    data['v_h'] = data['v'] / data['h']
    
    data['v_o'] = data['v'] / data['o']
