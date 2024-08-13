import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import trading_indicators as ti

# portfolio
portfolio = ['COIN', 'GRAB', 'INTC', 'MSFT', 'NVDA', 'SEAT', 'VTLE', 'GOOGL', 'IBM', 'NFLX', 'AMZN',
           'AMD', 'QQQ', 'QQQM', 'NEE', 'V', 'C', 'SQ', 'PHYS', 'SCHD', 'VOO', 'UNH', 'ABBV', 'HCA',
           'CI', 'LLY', 'ACIC', 'SBLK', 'SB', 'BP', 'VTLE', 'FSLR', 'CMG', 'JD', 'REAL', 'BABA',
           'AAP']

# dictionary to store historical data
stock_data = {}
for ticker in portfolio:
    stock = yf.Ticker(ticker)
    stock_data[ticker] = stock.history(period="2y")
    stock_data[ticker] = stock_data[ticker].drop(columns=['Volume', 'Dividends', 'Stock Splits'])

# convert dictionary to dataframe
portfolio_data = pd.concat(stock_data, axis=1)

# reindex all stock data to a common index and fill missing data
common_index = portfolio_data.index
for ticker in portfolio:
    stock_data[ticker] = stock_data[ticker].reindex(common_index).fillna(method='ffill')
    stock_data[ticker].dropna(inplace=True)

# plot closing data
plt.figure(figsize=(12, 7))

for ticker in portfolio:
    plt.plot(stock_data[ticker].index, stock_data[ticker]['Close'], label=f"{ticker} Closing Price")

plt.title("Closing Prices of Portfolio Stocks")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend(loc="best")
plt.show()

for ticker in portfolio:
    history = stock_data[ticker]

    # apply technical analysis
    ti.bollinger_bands(history)
    ti.rsi(history)
    ti.stochastic_oscillator(history)
    ti.MACD(history)
    ti.find_moving_averages(history)
    ti.find_trends(history)
    stock_data[ticker] = history

def generate_signals(df):
    # buy and sell signals
    df['Buy Signal'] = 0
    df['Sell Signal'] = 0

    # generate signals based on RSI and Bollinger Bands
    df.loc[(df['RSI'] < 30) & (df['Close'] < df['Lower Band']), 'Buy Signal'] = 1
    df.loc[(df['RSI'] > 70) & (df['Close'] > df['Upper Band']), 'Sell Signal'] = -1

for ticker in portfolio:
    generate_signals(stock_data[ticker])

def apply_stop_loss_take_profit(df, stop_loss=0.05, take_profit=0.10):
    df['Stop Loss'] = df['Close'] * (1 - stop_loss)
    df['Take Profit'] = df['Close'] * (1 + take_profit)
    df['Exit Signal'] = 0
    df.loc[df['Close'] <= df['Stop Loss'], 'Exit Signal'] = -1
    df.loc[df['Close'] >= df['Take Profit'], 'Exit Signal'] = 1

for ticker in portfolio:
    apply_stop_loss_take_profit(stock_data[ticker])

portfolio_returns = pd.DataFrame(index=stock_data[portfolio[0]].index)

def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()

for ticker in portfolio:
    history = stock_data[ticker]
    
    # calculate daily returns
    history['Daily Return'] = history['Close'].pct_change()
    
    # calculate strategy returns
    history['Strategy Return'] = history['Daily Return'] * history['Buy Signal'].shift(1)

    # calculate sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(history['Strategy Return'])
    
    # add to portfolio returns
    portfolio_returns[ticker] = history['Strategy Return']

# calculate overall portfolio return
portfolio_returns['Portfolio Return'] = portfolio_returns.mean(axis=1)
portfolio_returns['Cumulative Return'] = (1 + portfolio_returns['Portfolio Return']).cumprod()

# plot cumulative return
portfolio_returns['Cumulative Return'].plot(figsize=(12, 7), title="Portfolio Cumulative Return")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.show()