# -*- coding: utf-8 -*-
"""
@author: Ugur Okumus 
baseline code taken from Harnick Khera (Github.com/Hephyrius)

"""

import numpy as np
from numpy import *
import pandas as pd
import os

from binance.client import Client
from binance.enums import *

import CoreFunctions as cf

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from joblib import dump, load

#You don't need to enter your key/secret in order to get data from the exchange, its only needed for trades in the TradingBot.py class.
api_key = '0'
api_secret = '0'
client = Client(api_key, api_secret)

interval = Client.KLINE_INTERVAL_1DAY
start_date = "01 Jan, 2020"
end_date = "10 Jul, 2023"

interval_mapping = {item[0]: item[1] for item in cf.interval_list}

file_name = f"btc_usdt_{interval_mapping[interval]}.csv"
file_path = os.path.join("Data", file_name)
target_file_path = file_path.replace(".csv", "_targets.csv")

if os.path.exists(file_path) and os.path.exists(target_file_path):
    x = pd.read_csv(file_path)
    y = pd.read_csv(target_file_path)['target'].tolist()

else:
    # Fetch historical data
    candles = client.get_historical_klines("BTCUSDT", interval, start_date, end_date)
    
    # Convert the raw data from the exchange into a friendlier form with some basic feature creation
    x = cf.FeatureCreation(candles)

    # Create our targets
    y = cf.CreateTargets(candles, 1)

    #remove the top elements of the features and targets - this is for certain features that arent compatible with the top most
    #for example SMA27 would have 27 entries that would be incompatible/incomplete and would need to be discarded
    y = y[94:]
    x = x[94:len(candles)-1]

    # Save the feature dataframe and target list to CSV
    x.to_csv(file_path, index=False)
    pd.DataFrame(y, columns=['target']).to_csv(target_file_path, index=False)


print(x.head())
print(y[:10])

#produce sets, avoiding overlaps!
#data is seporated temporily rather than randomly
#this prevents the model learning stuff it wouldnt know - aka leakage - which can give us false positive models
#trny = y[:9999]
#trnx = x[:9999]

#Validation set is not used in this starter model, but should be used if using other libraries that support early stopping.
#valy = y[10000:12999]
#valx = x[10000:12999]

#tsty = y[13000:]
#tstx = x[13000:]

#model = GradientBoostingClassifier() 
#model.fit(trnx,trny)
#
#preds = model.predict(tstx)
#
##Some basic tests so we know how well our model performs on unseen - "modern" data.
##Helps with fine tuning features and model parameters
#accuracy = accuracy_score(tsty, preds)
#mse = mean_squared_error(tsty, preds)
#
#print("Accuracy = " + str(accuracy))
#print("MSE = " + str(mse))
#
#falsePos = 0
#falseNeg = 0
#truePos = 0
#trueNeg = 0
#total = len(preds)
#
#for i in range(len(preds)):
#    
#    if preds[i] == tsty[i] and tsty[i] == 1:
#        truePos +=1
#        
#    elif preds[i] == tsty[i] and tsty[i] == 0:
#        trueNeg +=1
#        
#    elif preds[i] != tsty[i] and tsty[i] == 1:
#        falsePos +=1
#        
#    elif preds[i] != tsty[i] and tsty[i] == 0:
#        falseNeg +=1
#        
#print("False Pos = " + str(falsePos/total))
#print("False Neg = " + str(falseNeg/total))
#print("True Pos = " + str(truePos/total))
#print("True Neg = " + str(trueNeg/total))
#
##how important of the features - helps with feature selection and creation!
#results = pd.DataFrame()
#results['names'] = trnx.columns
#results['importance'] = model.feature_importances_
#print(results.head)
#
#
##save our model to the system for use in the bot
#dump(model, open("Models/model.mdl", 'wb'))








