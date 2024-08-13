import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import trading_indicators as ti

# get data
tickers = ['COIN', 'GRAB', 'INTC', 'MSFT', 'NVDA', 'SEAT', 'VTLE', 'GOOGL', 'IBM', 'NFLX', 'AMZN',
           'AMD', 'QQQ', 'QQQM', 'NEE', 'V', 'C', 'SQ', 'PHYS', 'SCHD', 'VOO', 'UNH', 'ABBV', 'HCA',
           'CI', 'LLY', 'ACIC', 'SBLK', 'SB', 'BP', 'VTLE', 'FSLR', 'CMG', 'SBUX', 'JD', 'REAL', 'BABA',
           'AAP']
stock_data = {}
for ticker in tickers:
    stock = yf.Ticker(ticker).history(period="2y")
    stock_data[ticker] = stock.drop(columns=['Volume', 'Dividends', 'Stock Splits'])

# align all stock data to a common date index
common_index = pd.concat([stock_data[ticker] for ticker in tickers], axis=1).index
for ticker in tickers:
    stock_data[ticker] = stock_data[ticker].reindex(common_index).fillna(method='ffill')

# apply technical indicators
for ticker in tickers:
    ti.find_moving_averages(stock_data[ticker])
    ti.find_trends(stock_data[ticker])

# trend following strategy
def trend_following_strategy(df):
    df['Position'] = 0
    df.loc[(df['5d Trend'] > 0) & (df['20d Trend'] > 0), 'Position'] = 1
    df.loc[(df['5d Trend'] < 0) & (df['20d Trend'] < 0), 'Position'] = -1
    df['Daily Return'] = df['Close'].pct_change()
    df['Strategy Return'] = df['Daily Return'] * df['Position'].shift(1)
    df['Strategy Return'].fillna(0, inplace=True)
    return df

for ticker in tickers:
    stock_data[ticker] = trend_following_strategy(stock_data[ticker])

# backtesting
portfolio_returns = pd.DataFrame({ticker: stock_data[ticker]['Strategy Return'] for ticker in tickers})
portfolio_returns['Portfolio Return'] = portfolio_returns.mean(axis=1).fillna(0)
portfolio_returns['Cumulative Return'] = (1 + portfolio_returns['Portfolio Return']).cumprod()

# plot cumulative return
plt.figure(figsize=(12, 7))
portfolio_returns['Cumulative Return'].plot(title="Trend-Following Strategy Cumulative Return")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")

# calculate Sharpe ratio
sharpe_ratio = np.sqrt(252) * (portfolio_returns['Portfolio Return'].mean() - 0.01 / 252) / portfolio_returns['Portfolio Return'].std()
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# calculate and plot max drawdown
drawdown = (portfolio_returns['Cumulative Return'] - portfolio_returns['Cumulative Return'].cummax()) / portfolio_returns['Cumulative Return'].cummax()
max_drawdown = drawdown.min()
print(f"Maximum Drawdown: {max_drawdown:.2%}")

plt.figure(figsize=(12, 7))
drawdown.plot(title="Portfolio Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown")

# show the plots
plt.show()
