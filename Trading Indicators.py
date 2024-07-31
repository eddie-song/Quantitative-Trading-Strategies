# this file is for testing indicators on specific stocks

import yfinance as yf
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
def calculate_bollinger_signals(df):
    # create the signal column
    df['Bollinger Signal'] = 0

    # set signals
    df.loc[df['Close'] < df['Lower Band'], 'Bollinger Signal'] = 1
    df.loc[df['Close'] > df['Upper Band'], 'Bollinger Signal'] = -1

calculate_bollinger_signals(history)

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

def find_trends(df):
    # find 5d trend
    rolling_gain = df['Gain'].rolling(window=5).sum()
    rolling_loss = df['Loss'].rolling(window=5).sum()
    df['5d Trend'] = rolling_gain - rolling_loss
    # df['5d Trend'] = (rolling_gain - rolling_loss).apply(lambda x: 1 if x > 0 else -1)

    # find 20d trend
    rolling_gain = df['Gain'].rolling(window=20).sum()
    rolling_loss = df['Loss'].rolling(window=20).sum()
    df['20d Trend'] = rolling_gain - rolling_loss
    # df['20d Trend'] = (rolling_gain - rolling_loss).apply(lambda x: 1 if x > 0 else -1)

    # find 200d trend
    rolling_gain = df['Gain'].rolling(window=200).sum()
    rolling_loss = df['Loss'].rolling(window=200).sum()
    df['200d Trend'] = rolling_gain - rolling_loss
    # df['200d Trend'] = (rolling_gain - rolling_loss).apply(lambda x: 1 if x > 0 else -1)
    
find_trends(history)

# def rsi_and_macd_signals(df):

history = history.tail(50)
history.to_csv("{} CSV File".format(ticker), index=False, sep='\t')
history.to_markdown("{} Table File".format(ticker), index=False)

def plot_bands(df):
    plt.figure(figsize=(12,7))

    # plot the closing prices
    plt.plot(df['Close'], label = "Closing Price", color = 'b')

    # plot the bollinger bands
    plt.plot(df['Upper Band'], label = "Upper Bollinger Band", color='r')
    plt.plot(df['Lower Band'], label = "Lower Bollinger Band", color='r')

    # plot the signals
    plt.plot(df[df['Bollinger Signal'] == 1].index, df['Lower Band'][df['Bollinger Signal'] == 1], '^', markersize=10, color='g', lw=0, label='Buy Signal')
    plt.plot(df[df['Bollinger Signal'] == -1].index, df['Upper Band'][df['Bollinger Signal'] == -1], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

    plt.title("{} Bollinger Bonds".format(ticker))
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="best")

def plot_smas(df):
    plt.figure(figsize=(12,7))

    # plot the moving averages
    plt.plot(df['20d Moving Average'], label = "20d Moving Average", color = 'g', linestyle='-')
    plt.plot(df['12d EMA'], label = "12d Exponential Moving Average", color = 'g', linestyle='--')
    plt.plot(df['26d EMA'], label = "26d Exponential Moving Average", color = 'g', linestyle=':')

    plt.title("{} SMAs".format(ticker))
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="best")

def plot_rsi(df):
    plt.figure(figsize=(12,7))

    # plot the rsi
    plt.plot(df['RSI'], label="Relative Strength Index")
    plt.axhline(y = 70, color = 'r', linestyle=':', label="Normal Overbought Indicator")
    plt.axhline(y = 30, color = 'r', linestyle='--', label="Normal Oversold Indicator")

    plt.title("{} Relative Strength Index".format(ticker))
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend(loc="best")

def plot_stochastic_oscillator(df):
    plt.figure(figsize=(12,7))

    # plot the stochastic oscillator
    plt.plot(df['%K'], label="%K")
    plt.plot(df['%D'], label="%D")
    plt.axhline(y = 80, color = 'r', linestyle=':', label="Overbought Indicator")
    plt.axhline(y = 20, color = 'r', linestyle='--', label="Oversold Indicator")

    plt.legend(loc="best")
    plt.title("{} Stochastic Oscillator".format(ticker))
    plt.xlabel("Date")
    plt.ylabel("Stochastic Oscillator Value")

def plot_macd(df):
    plt.figure(figsize=(12,7))

    # plot the MACD
    plt.plot(df['MACD'], label="MACD")
    plt.plot(df['Signal Line'], label="Signal Line")
    # plt.hist(history['MACD and Signal Line Difference'])
    plt.axhline(y = 0, color='k')

    plt.legend(loc="best")
    plt.title("{} MACD".format(ticker))
    plt.xlabel("Date")
    plt.ylabel("Value")

    plt.figure(figsize=(12,7))

    # plot the difference between MACD and signal line
    plt.bar(df.index, df['MACD and Signal Line Difference'], label='MACD and Signal Line Difference')
    plt.axhline(y = 0, color='k')

    plt.legend(loc="best")
    plt.title("{} MACD".format(ticker))
    plt.xlabel("Date")
    plt.ylabel("Value")

def plot_trends(df):
    plt.figure(figsize=(15,7))

    # plot the trends
    temp = df[df['5d Trend'] != 0]
    plt.bar(df.index, df['5d Trend'], color='blue', alpha=0.5, label='5d Trend')
    plt.bar(df.index, df['20d Trend'], color='green', alpha=0.5, label='20d Trend')
    plt.bar(df.index, df['200d Trend'], color='red', alpha=0.5, label='200d Trend')
    plt.axhline(y = 0, color='k')
    # plt.ylim(-3, 3)

    plt.legend(loc="best")
    plt.title("{} Overall Trends".format(ticker))
    plt.xlabel("Date")

# plot_bands(history)
# plot_trends(history)
plot_macd(history)

# display the plots
plt.show()

