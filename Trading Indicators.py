import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# select stock
ticker="GOOGL"
stock = yf.Ticker(ticker)
history = stock.history(period="2y") # historical prices of stock
n = 20 # number of days in smoothing period
m = 2 # number of standard deviations
history = history.drop(columns = ['Volume', 'Dividends', 'Stock Splits']) # drop unecessary columns

# bollinger bands
def bollinger_bands(df):
    # calculate the typical price
    df['Typical Price'] = (df['High'] + df['Low'] + df['Close']) / 3

    # calculate the SMA
    df['20d Moving Average'] = df['Typical Price'].rolling(window = n).mean()
    df['20d Moving Std Dev'] = df['Typical Price'].rolling(window = n).std()

    # calculate the band values
    df['Upper Band'] = df['20d Moving Average'] + (m * df['20d Moving Std Dev'])
    df['Lower Band'] = df['20d Moving Average'] - (m * df['20d Moving Std Dev'])

bollinger_bands(history)

# find the signals
def calculate_signals(df):
    # create the signal column
    df['Signal'] = 0

    # set signals
    df.loc[df['Close'] < df['Lower Band'], 'Signal'] = 1
    df.loc[df['Close'] > df['Upper Band'], 'Signal'] = -1

calculate_signals(history)

# relative strength index
def rsi(df):
    # create gain and loss columns
    df['Gain'] = 0
    df['Loss'] = 0

    for i in range(1, len(df)):
        # calculate the change in price
        change = df['Close'].iloc[i] - df['Close'].iloc[i-1]

        # add to respective column
        if change > 0:
            df['Gain'].iloc[i] = change
        else:
            df['Loss'].iloc[i] = abs(change)
    
    # calculate the average gains and losses
    df['Avg. Gains'] = df['Gain'].rolling(window=14).mean()
    df['Avg. Loss'] = df['Loss'].rolling(window=14).mean()

    # calculate the RS and RSI
    df['RS'] = df['Avg. Gains'] / df['Avg. Loss']
    df['RSI'] = 100 - (100 / (1 + df['RS']))

rsi(history)

# stochastic oscillator
def stochastic_oscillator(df):
    # find the 14 day high and low
    df['14d High'] = df['Close'].rolling(window=14).max()
    df['14d Low'] = df['Close'].rolling(window=14).min()

    # calculate the slow and fast stochastic oscillators
    df['%K'] = ((df['Close'] - df['14d Low']) / (df['14d High'] - df['14d Low'])) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()

stochastic_oscillator(history)

# moving average convergence divergence
def MACD(df):
    # calculate the 12 day and 26 day EMAs
    df['12d EMA'] = df['Close'].ewm(span=12).mean()
    df['26d EMA'] = df['Close'].ewm(span=26).mean()

    # calculate the MACD
    df['MACD'] = df['12d EMA'] - df['26d EMA']

    # calculate the signal line
    df['Signal Line'] = df['MACD'].ewm(span=9).mean()

    # calculate the difference between the MACD and signal line
    df['MACD and Signal Line Difference'] = df['MACD'] - df['Signal Line']

MACD(history)

history = history.tail(50)
history.to_csv("{} CSV File".format(ticker), index=False, sep='\t')

# plot the closing prices
plt.figure(figsize=(12,7))
plt.plot(history['Close'], label = "Closing Price", color = 'b')

# plot the moving averages
plt.plot(history['20d Moving Average'], label = "20d Moving Average", color = 'g', linestyle='-')
plt.plot(history['12d EMA'], label = "12d Exponential Moving Average", color = 'g', linestyle='--')
plt.plot(history['26d EMA'], label = "26d Exponential Moving Average", color = 'g', linestyle=':')

# plot the bollinger bands
plt.plot(history['Upper Band'], label = "Upper Bollinger Band", color='r')
plt.plot(history['Lower Band'], label = "Lower Bollinger Band", color='r')

# plot the signals
plt.plot(history[history['Signal'] == 1].index, history['Close'][history['Signal'] == 1], '^', markersize=10, color='g', lw=0, label='Buy Signal')
plt.plot(history[history['Signal'] == -1].index, history['Close'][history['Signal'] == -1], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

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

# plot the stochastic oscillator
plt.figure(figsize=(12,7))
plt.plot(history['%K'], label="%K")
plt.plot(history['%D'], label="%D")
plt.axhline(y = 80, color = 'r', linestyle=':', label="Overbought Indicator")
plt.axhline(y = 20, color = 'r', linestyle='--', label="Oversold Indicator")

plt.legend(loc="best")
plt.title("{} Stochastic Oscillator".format(ticker))
plt.xlabel("Date")
plt.ylabel("Stochastic Oscillator Value")

# plot the MACD
plt.figure(figsize=(12,7))
plt.plot(history['MACD'], label="MACD")
plt.plot(history['Signal Line'], label="Signal Line")
# plt.hist(history['MACD and Signal Line Difference'])
plt.axhline(y = 0, color='k')

plt.legend(loc="best")
plt.title("{} MACD".format(ticker))
plt.xlabel("Date")
plt.ylabel("Value")

# display the plot
plt.show()

