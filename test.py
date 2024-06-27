import DataPreprocessor as dp
import pandas as pd
# Örnek kullanım:
interval = '1d'
start_date = "01 Jan, 2021"
end_date = "10 Jul, 2023"
x, y = dp.get_historical_data(interval, start_date, end_date) 

print(x.shape)
print(y.shape)

#x = dp.get_latest_data(interval, limit=1000)
#x['timestamp'] = pd.to_datetime(x['timestamp'], unit='s')
#print(x[['timestamp', 'open', 'high', 'low', 'close']])