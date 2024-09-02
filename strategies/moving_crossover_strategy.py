import trading_indicators as ti
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# select stock
ticker="NVDA"
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

def calculate_cagr(initial_value, final_value, period_in_years):
    return (final_value / initial_value) ** (1 / period_in_years) - 1

def calculate_max_drawdown(portfolio_values):
    peak = portfolio_values[0]
    max_drawdown = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown

def backtest_moving_crossover_strategy(df):
    data = df.copy()
    initial_cash = 10000
    cash = initial_cash
    position = 0
    portfolio_values = []
    trade_signals = []
    winning_trades = 0
    losing_trades = 0
    time_in_market = 0

    for i in range(1, len(data)):
        # check for buy signal
        if data['50-200 Position'].iloc[i] == 1 or data['20-50 Position'].iloc[i] == 1:
            if position == 0:  # Buy if no current position
                position = cash / data['Close'].iloc[i]
                cash = 0
                trade_signals.append((data.index[i], 'Buy', data['Close'].iloc[i]))
                print(f"Bought on {data.index[i]} at price {data['Close'].iloc[i]}")
        
        # check for sell signal
        elif data['50-200 Position'].iloc[i] == -1 or data['20-50 Position'].iloc[i] == -1:
            if position > 0:  # Sell if in position
                cash = position * data['Close'].iloc[i]
                trade_signals.append((data.index[i], 'Sell', data['Close'].iloc[i]))
                print(f"Sold on {data.index[i]} at price {data['Close'].iloc[i]}")
                if trade_signals[-2][2] < data['Close'].iloc[i]:
                    winning_trades += 1
                else:
                    losing_trades += 1
                position = 0

        # track portfolio value
        if position > 0:
            portfolio_value = position * data['Close'].iloc[i]
            time_in_market += 1  # Increase time in market when in position
        else:
            portfolio_value = cash
        portfolio_values.append(portfolio_value)

    # calculate final portfolio value
    final_value = portfolio_values[-1]
    strategy_return = (final_value - initial_cash) / initial_cash

    # calculate benchmark returns
    buy_and_hold_final_value = initial_cash * data['Close'].iloc[-1] / data['Close'].iloc[0]
    buy_and_hold_return = (buy_and_hold_final_value - initial_cash) / initial_cash

    # calculate CAGR
    period_in_years = len(data) / 252  # Approximate trading days in a year
    strategy_cagr = calculate_cagr(initial_cash, final_value, period_in_years)
    buy_and_hold_cagr = calculate_cagr(initial_cash, buy_and_hold_final_value, period_in_years)

    # calculate max drawdown
    strategy_max_drawdown = calculate_max_drawdown(portfolio_values)
    buy_and_hold_max_drawdown = calculate_max_drawdown(
        [initial_cash * price / data['Close'].iloc[0] for price in data['Close']]
    )

    # calcalate winrate
    total_trades = winning_trades + losing_trades
    winrate = winning_trades / total_trades if total_trades > 0 else 0
    time_in_market_percent = (time_in_market / len(data)) * 100

    # print results
    print(f"Strategy return: {strategy_return * 100:.2f}%")
    print(f"Buy and hold return: {buy_and_hold_return * 100:.2f}%")
    print(f"Strategy CAGR: {strategy_cagr * 100:.2f}%")
    print(f"Buy and hold CAGR: {buy_and_hold_cagr * 100:.2f}%")
    print(f"Strategy Max Drawdown: {strategy_max_drawdown * 100:.2f}%")
    print(f"Buy and hold Max Drawdown: {buy_and_hold_max_drawdown * 100:.2f}%")
    print(f"Winning trades: {winning_trades}")
    print(f"Losing trades: {losing_trades}")
    print(f"Total trades: {total_trades}")
    print(f"Time in market: {time_in_market} days ({time_in_market_percent:.2f}%)")
    print(f"Winrate: {winrate * 100:.2f}%")

moving_crossover_strategy(history)
backtest_moving_crossover_strategy(history)