import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date as dt
from dateutil.relativedelta import relativedelta

# select stock
ticker="JNJ"
stock = yf.Ticker(ticker)
history = stock.history(period="2y") # historical prices of stock
n = 20 # number of days in smoothing period
m = 2 # number of standard deviations
history = history.drop(columns = ['Volume', 'Dividends', 'Stock Splits'])

# bollinger bands
def bollinger_bands(df):
    df['Typical Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['20d Moving Average'] = df['Typical Price'].rolling(window = n).mean()
    df['20d Moving Std Dev'] = df['Typical Price'].rolling(window = n).std()
    df['Upper Band'] = df['20d Moving Average'] + (m * df['20d Moving Std Dev'])
    df['Lower Band'] = df['20d Moving Average'] - (m * df['20d Moving Std Dev'])

bollinger_bands(history)

# find the signals
def calculate_signals(df):
    df['Signal'] = 0
    df.loc[df['Close'] < df['Lower Band'], 'Signal'] = 1
    df.loc[df['Close'] > df['Upper Band'], 'Signal'] = -1

calculate_signals(history)

# relative strength index
def rsi(df):
    df['Gain'] = 0
    df['Loss'] = 0
    for i in range(1, len(df)):
        change = df['Close'].iloc[i] - df['Close'].iloc[i-1]
        if change > 0:
            df['Gain'].iloc[i] = change
        else:
            df['Loss'].iloc[i] = abs(change)
    
    df['Avg. Gains'] = df['Gain'].rolling(window=14).mean()
    df['Avg. Loss'] = df['Loss'].rolling(window=14).mean()
    df['RS'] = df['Avg. Gains'] / df['Avg. Loss']
    df['RSI'] = 100 - (100 / (1 + df['RS']))


rsi(history)

# plot the closing prices
plt.figure(figsize=(12,7))
plt.plot(history['Close'], label = "Closing Price", color = 'b')

# plot the bollinger bands
plt.plot(history['20d Moving Average'], label = "20d Moving Average", color = 'g')
plt.plot(history['Upper Band'], label = "Upper Bollinger Band", color='r')
plt.plot(history['Lower Band'], label = "Lower Bollinger Band", color='r')

# plot the signals
plt.plot(history[history['Signal'] == 1].index, history['Lower Band'][history['Signal'] == 1], '^', markersize=10, color='g', lw=0, label='Buy Signal')
plt.plot(history[history['Signal'] == -1].index, history['Upper Band'][history['Signal'] == -1], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

plt.title("{} Mean Reversion Strategy".format(ticker))
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend(loc="best")

# plot the rsi
plt.figure(figsize=(12,7))
plt.plot(history['RSI'], label="Relative Strength Index")
plt.axhline(y = 70, color = 'r', linestyle=':', label="Normal Overbought Indicator")
plt.axhline(y = 30, color = 'r', linestyle='--', label="Normal Oversold Indicator")

plt.title("{} Relative Strength Index".format(ticker))
plt.xlabel("Date")
plt.ylabel("RSI")
plt.legend(loc="best")

# display the plot
plt.show()

