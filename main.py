import os
from dotenv import load_dotenv
import ccxt

# .env dosyasındaki değişkenleri yükleyin
load_dotenv()

# Çevresel değişkenlerden API anahtarlarını alın
api_key = os.getenv('BINANCE_API_KEY')
secret_key = os.getenv('BINANCE_SECRET_KEY')

# Binance API ile iletişim kurun
binance = ccxt.binance({
    'apiKey': api_key,
    'secret': secret_key,
})

# Örneğin BTC/USDT fiyatlarını çekme
ticker = binance.fetch_ticker('BTC/USDT')
#print(ticker)


exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1m'  # 1 dakika
limit = 500  # 500 mum

candles = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
print(candles)
