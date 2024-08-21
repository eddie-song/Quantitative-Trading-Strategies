import trading_indicators as ti
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# select stock
ticker="BP"
stock = yf.Ticker(ticker)
history = stock.history(period="2y") # historical prices of stock

# drop unecessary columns
history = history.drop(columns = ['Volume', 'Dividends', 'Stock Splits'])

def moving_crossover_strategy(df):
    data = df.copy() # copy of dataframe to be manipulated
    data['Signal'] = 0 # create signal column
    ti.find_moving_averages(data)

    # generate signals for 50 and 200 day moving averages
    data['50-200 Signal'] = 0.0
    data['50-200 Signal'][50:] = np.where(data['50d Moving Average'][50:] > data['200d Moving Average'][50:], 1.0, 0.0)

    data['50-200 Position'] = data['50-200 Signal'].diff()

    # generate signals for 20 and 50 day moving averages
    data['20-50 Signal'] = 0.0
    data['20-50 Signal'][20:] = np.where(data['20d Moving Average'][20:] > data['50d Moving Average'][20:], 1.0, 0.0)

    data['20-50 Position'] = data['20-50 Signal'].diff()
    
    ti.plot_moving_averages(data)

    # plot signals
    plt.plot(data[data['50-200 Position'] == 1].index, data['50d Moving Average'][data['50-200 Position'] == 1], '^', markersize=10, color='g', lw=0, label='Buy Signal')
    plt.plot(data[data['50-200 Position'] == -1].index, data['200d Moving Average'][data['50-200 Position'] == -1], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

    plt.plot(data[data['20-50 Position'] == 1].index, data['20d Moving Average'][data['20-50 Position'] == 1], '^', markersize=10, color='g', lw=0, label='Buy Signal')
    plt.plot(data[data['20-50 Position'] == -1].index, data['50d Moving Average'][data['20-50 Position'] == -1], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

    plt.show()

moving_crossover_strategy(history)
