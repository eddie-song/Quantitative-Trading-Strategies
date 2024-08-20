# trading indicators to be used in strategies

import yfinance as yf
import matplotlib.pyplot as plt

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# # select stock
ticker="BP"
# stock = yf.Ticker(ticker)
# history = stock.history(period="2y") # historical prices of stock
n = 20 # number of days in smoothing period
m = 2 # number of standard deviations
# history = history.drop(columns = ['Volume', 'Dividends', 'Stock Splits']) # drop unecessary columns

# universal methods
# find gain/loss and average gain/loss
def find_gains_and_losses(df):
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
    
    # calculate the average gains and losses over the previous 14 days
    df['Avg. Gains'] = df['Gain'].rolling(window=14).mean()
    df['Avg. Loss'] = df['Loss'].rolling(window=14).mean()


# indicators

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

# find the signals
def calculate_bollinger_signals(df):
    # create the signal column
    df['Bollinger Signal'] = 0

    # set signals
    df.loc[df['Close'] < df['Lower Band'], 'Bollinger Signal'] = 1
    df.loc[df['Close'] > df['Upper Band'], 'Bollinger Signal'] = -1


# relative strength index
def rsi(df):
    find_gains_and_losses(df)
    # calculate the RS and RSI
    df['RS'] = df['Avg. Gains'] / df['Avg. Loss']
    df['RSI'] = 100 - (100 / (1 + df['RS']))


# stochastic oscillator
def stochastic_oscillator(df):
    # find the 14 day high and low
    df['14d High'] = df['Close'].rolling(window=14).max()
    df['14d Low'] = df['Close'].rolling(window=14).min()

    # calculate the slow and fast stochastic oscillators
    df['%K'] = ((df['Close'] - df['14d Low']) / (df['14d High'] - df['14d Low'])) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()


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


# trends
def find_trends(df):
    find_gains_and_losses(df)
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


# moving averages
def find_moving_averages(df):
    # calculate the 50 day moving average
    df['50d Moving Average'] = df['Close'].rolling(window=50).mean()

    # calculate the 200 day moving average
    df['200d Moving Average'] = df['Close'].rolling(window=200).mean()

# plot indicators
def plot_bands(df):
    bollinger_bands(df)
    calculate_bollinger_signals(df)
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

def plot_emas(df):
    MACD(df)
    plt.figure(figsize=(12,7))

    # plot the closing price
    plt.plot(df['Close'], label = "Closing Price", color = 'b')

    # plot the moving averages
    plt.plot(df['12d EMA'], label = "12d Exponential Moving Average", color = 'g', linestyle='--')
    plt.plot(df['26d EMA'], label = "26d Exponential Moving Average", color = 'g', linestyle=':')

    plt.title("{} EMAs".format(ticker))
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="best")

def plot_rsi(df):
    rsi(df)
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
    stochastic_oscillator(df)
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
    MACD(df)
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
    find_trends(df)
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

def plot_moving_averages(df):
    find_moving_averages(df)
    plt.figure(figsize=(12,7))

    # plot the closing prices
    plt.plot(df['Close'], label = "Closing Price", color = 'b')

    # plot the moving averages
    plt.plot(df['20d Moving Average'], label = "20d Moving Average", color = 'k')
    plt.plot(df['50d Moving Average'], label = "50d Moving Average", color = 'g')
    plt.plot(df['200d Moving Average'], label = "200d Moving Average", color = 'r')

    plt.title("{} Moving Averages".format(ticker))
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="best")

# convert to csv and markdown
# history.to_csv("./CSV Files/{} CSV File".format(ticker), index=False, sep='\t')
# history.to_markdown("./Table Files/{} Table File".format(ticker), index=False)

# plot_bands(history)
# plot_stochastic_oscillator(history)
# plot_moving_averages(history)
# plot_emas(history)
# plot_rsi(history)
# plot_trends(history)
# plot_macd(history)

# display the plots
# plt.show()

