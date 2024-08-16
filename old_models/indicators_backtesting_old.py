import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import trading_indicators as ti
import numpy as np

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# portfolio
portfolio = ['COIN', 'GRAB', 'INTC', 'MSFT', 'NVDA', 'SEAT', 'VTLE', 'GOOGL', 'IBM', 'NFLX', 'AMZN',
           'AMD', 'QQQ', 'QQQM', 'NEE', 'V', 'C', 'SQ', 'PHYS', 'SCHD', 'VOO', 'UNH', 'ABBV', 'HCA',
           'CI', 'LLY', 'ACIC', 'SBLK', 'SB', 'BP', 'VTLE', 'FSLR', 'CMG', 'JD', 'REAL', 'BABA',
           'AAP']

# download data for each stock
stock_data = {}
for ticker in portfolio:
    stock = yf.Ticker(ticker)
    stock_data[ticker] = stock.history(period="2y")
    stock_data[ticker] = stock_data[ticker].drop(columns=['Volume', 'Dividends', 'Stock Splits'])

# convert to df
portfolio_data = pd.concat(stock_data, axis=1)

# reindex data and drop null values
common_index = portfolio_data.index
for ticker in portfolio:
    stock_data[ticker] = stock_data[ticker].reindex(common_index).fillna(method='ffill')
    stock_data[ticker].dropna(inplace=True)

print(stock_data)

# plot closing data
plt.figure(figsize=(12, 7))
for ticker in portfolio:
    plt.plot(stock_data[ticker].index, stock_data[ticker]['Close'], label=f"{ticker} Closing Price")
plt.title("Closing Prices of Portfolio Stocks")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend(loc="best")
plt.show()

# apply technical analysis
for ticker in portfolio:
    history = stock_data[ticker]

    ti.bollinger_bands(history)
    ti.rsi(history)
    ti.stochastic_oscillator(history)
    ti.MACD(history)
    ti.find_moving_averages(history)
    ti.find_trends(history)
    stock_data[ticker] = history

# RSI and Bollinger Bands backtesting
def rsi_and_bollinger_bands_backtest(df):
    df['Buy Signal'] = 0
    df['Sell Signal'] = 0

    df.loc[(df['RSI'] < 30) & (df['Close'] < df['Lower Band']), 'Buy Signal'] = 1
    df.loc[(df['RSI'] > 70) & (df['Close'] > df['Upper Band']), 'Sell Signal'] = -1

for ticker in portfolio:
    rsi_and_bollinger_bands_backtest(stock_data[ticker])

# apply stop-loss and take-profit rules
def apply_stop_loss_take_profit(df, stop_loss=0.05, take_profit=0.10):
    df['Stop Loss'] = df['Close'] * (1 - stop_loss)
    df['Take Profit'] = df['Close'] * (1 + take_profit)
    df['Exit Signal'] = 0
    df.loc[df['Close'] <= df['Stop Loss'], 'Exit Signal'] = -1
    df.loc[df['Close'] >= df['Take Profit'], 'Exit Signal'] = 1

for ticker in portfolio:
    apply_stop_loss_take_profit(stock_data[ticker])

# calculate returns
portfolio_returns = pd.DataFrame(index=stock_data[portfolio[0]].index)

# calculate Sharpe Ratio
def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()

# calculate daily and strategy returns
for ticker in portfolio:
    history = stock_data[ticker]
    
    history['Daily Return'] = history['Close'].pct_change()
    history['Strategy Return'] = history['Daily Return'] * history['Buy Signal'].shift(1)
    sharpe_ratio = calculate_sharpe_ratio(history['Strategy Return'])
    
    portfolio_returns[ticker] = history['Strategy Return']

# calculate overall portfolio return
portfolio_returns['Portfolio Return'] = portfolio_returns.mean(axis=1)
portfolio_returns['Cumulative Return'] = (1 + portfolio_returns['Portfolio Return']).cumprod()

# plot cumulative return
portfolio_returns['Cumulative Return'].plot(figsize=(12, 7), title="Portfolio Cumulative Return")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.show()

# moving average crossover
def moving_average_crossover(df, short_window=50, long_window=200):
    df['Short MA'] = df['Close'].rolling(window=short_window).mean()
    df['Long MA'] = df['Close'].rolling(window=long_window).mean()
    df['MA Crossover Signal'] = 0
    
    # generate signals
    df['Signal'] = np.where(df['Short MA'] > df['Long MA'], 1, -1)
    
    # filter the signals to avoid clustering
    df['MA Crossover Signal'] = df['Signal'].diff()
    
    # filter out consecutive same signals
    df['Buy Signal'] = np.where(df['MA Crossover Signal'] > 0, 1, 0)
    df['Sell Signal'] = np.where(df['MA Crossover Signal'] < 0, -1, 0)

for ticker in portfolio:
    moving_average_crossover(stock_data[ticker])

# plot moving average crossover signals
def plot_moving_average_crossover(df, ticker):
    plt.figure(figsize=(12, 7))
    plt.plot(df['Close'], label="Close Price", color='black')
    plt.plot(df['Short MA'], label="50-Day MA", color='blue')
    plt.plot(df['Long MA'], label="200-Day MA", color='red')

    plt.plot(df[df['Buy Signal'] == 1].index,
             df['Short MA'][df['Buy Signal'] == 1],
             '^', markersize=10, color='g', lw=0, label='Buy Signal')

    plt.plot(df[df['Sell Signal'] == -1].index,
             df['Short MA'][df['Sell Signal'] == -1],
             'v', markersize=10, color='r', lw=0, label='Sell Signal')

    plt.title(f"{ticker} Moving Average Crossover Strategy")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="best")
    plt.show()

# apply the function
ticker = 'MSFT'
plot_moving_average_crossover(stock_data[ticker], ticker)

# momentum strategy
def momentum_strategy(df, lookback_period=252):
    # calculate the momentum
    df['Momentum'] = df['Close'].pct_change(periods=lookback_period)
    
    # generate initial signals based on momentum
    df['Signal'] = np.where(df['Momentum'] > 0, 1, -1)
    
    # create a momentum crossover signals
    df['Momentum Signal'] = df['Signal'].diff()
    
    # filter out consecutive same signals
    df['Buy Signal'] = np.where(df['Momentum Signal'] > 0, 1, 0)
    df['Sell Signal'] = np.where(df['Momentum Signal'] < 0, -1, 0)

for ticker in portfolio:
    momentum_strategy(stock_data[ticker])

# plot momentum signals
def plot_momentum_strategy(df, ticker):
    plt.figure(figsize=(12, 7))
    plt.plot(df['Close'], label="Close Price", color='black')

    plt.plot(
        df[df['Buy Signal'] == 1].index,
        df['Close'][df['Buy Signal'] == 1],
        '^', markersize=10, color='g', lw=0, label='Buy Signal'
    )
    plt.plot(
        df[df['Sell Signal'] == -1].index,
        df['Close'][df['Sell Signal'] == -1],
        'v', markersize=10, color='r', lw=0, label='Sell Signal'
    )

    plt.title(f"{ticker} Momentum Strategy")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="best")
    plt.show()

# apply the function
ticker = 'NVDA'
plot_momentum_strategy(stock_data[ticker], ticker)

# calculate Sharpe Ratio for Moving Average Crossover Strategy
for ticker in portfolio:
    history = stock_data[ticker]
    history['Strategy Return MA'] = history['Daily Return'] * history['Buy Signal'].shift(1)
    sharpe_ratio_ma = calculate_sharpe_ratio(history['Strategy Return MA'])
    print(f"{ticker} Sharpe Ratio (MA Crossover): {sharpe_ratio_ma}")

# calculate Sharpe Ratio for Momentum Strategy
for ticker in portfolio:
    history = stock_data[ticker]
    history['Strategy Return Momentum'] = history['Daily Return'] * history['Buy Signal'].shift(1)
    sharpe_ratio_momentum = calculate_sharpe_ratio(history['Strategy Return Momentum'])
    print(f"{ticker} Sharpe Ratio (Momentum): {sharpe_ratio_momentum}")

# calculate overall portfolio return for the new strategies
portfolio_returns['Portfolio Return MA'] = portfolio_returns.mean(axis=1)
portfolio_returns['Cumulative Return MA'] = (1 + portfolio_returns['Portfolio Return MA']).cumprod()

portfolio_returns['Portfolio Return Momentum'] = portfolio_returns.mean(axis=1)
portfolio_returns['Cumulative Return Momentum'] = (1 + portfolio_returns['Portfolio Return Momentum']).cumprod()

# plot cumulative return for the new strategies
portfolio_returns['Cumulative Return MA'].plot(figsize=(12, 7), title="Portfolio Cumulative Return (MA Crossover)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.show()

portfolio_returns['Cumulative Return Momentum'].plot(figsize=(12, 7), title="Portfolio Cumulative Return (Momentum)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.show()
