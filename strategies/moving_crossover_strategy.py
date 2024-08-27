import trading_indicators as ti
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# select stock
ticker="GOOGL"
stock = yf.Ticker(ticker)
history = stock.history(period="2y") # historical prices of stock

# drop unecessary columns
history = history.drop(columns = ['Volume', 'Dividends', 'Stock Splits'])

def moving_crossover_strategy(df):
    df['Signal'] = 0 # create signal column
    ti.find_moving_averages(df)

    # generate signals for 50 and 200 day moving averages
    df['50-200 Signal'] = 0.0
    df['50-200 Signal'][50:] = np.where(df['50d Moving Average'][50:] > df['200d Moving Average'][50:], 1.0, 0.0)

    df['50-200 Position'] = df['50-200 Signal'].diff()

    # generate signals for 20 and 50 day moving averages
    df['20-50 Signal'] = 0.0
    df['20-50 Signal'][20:] = np.where(df['20d Moving Average'][20:] > df['50d Moving Average'][20:], 1.0, 0.0)

    df['20-50 Position'] = df['20-50 Signal'].diff()
    
    ti.plot_moving_averages(df)

    # plot signals
    plt.plot(df[df['50-200 Position'] == 1].index, df['50d Moving Average'][df['50-200 Position'] == 1], '^', markersize=10, color='g', lw=0, label='Buy Signal')
    plt.plot(df[df['50-200 Position'] == -1].index, df['200d Moving Average'][df['50-200 Position'] == -1], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

    plt.plot(df[df['20-50 Position'] == 1].index, df['20d Moving Average'][df['20-50 Position'] == 1], '^', markersize=10, color='g', lw=0, label='Buy Signal')
    plt.plot(df[df['20-50 Position'] == -1].index, df['50d Moving Average'][df['20-50 Position'] == -1], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

    plt.show()

def backtest_moving_crossover_strategy(df):
    data = df.copy()
    initial_cash = 10000
    cash = initial_cash
    position = 0
    returns = []

    for i in range(1, len(data)):
        # check for buy signal
        if data['50-200 Position'].iloc[i] == 1 or data['20-50 Position'].iloc[i] == 1:
            # buy the stock if there's a buy signal and no current position
            if position == 0:
                position = cash / data['Close'].iloc[i]
                cash = 0
                print(f"Bought on {data.index[i]} at price {data['Close'].iloc[i]}")

        # check for sell signal
        elif data['50-200 Position'].iloc[i] == -1 or data['20-50 Position'].iloc[i] == -1:
            # sell the stock if there's a sell signal and a current position
            if position > 0:
                cash = position * data['Close'].iloc[i]
                position = 0
                print(f"Sold on {data.index[i]} at price {data['Close'].iloc[i]}")
                returns.append(cash)

    # calculate final portfolio value
    if position > 0:
        final_value = position * data['Close'].iloc[-1]
    else:
        final_value = cash

    # calculate strategy return
    strategy_return = (final_value - initial_cash) / initial_cash
    buy_and_hold_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]

    print(f"Strategy return: {strategy_return * 100:.2f}%")
    print(f"Buy and hold return: {buy_and_hold_return * 100:.2f}%")

    return strategy_return, buy_and_hold_return

moving_crossover_strategy(history)
backtest_moving_crossover_strategy(history)
