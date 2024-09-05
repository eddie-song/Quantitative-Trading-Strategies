import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# capital asset pricing model

# get data
stock_symbol = 'GOOGL'
market_symbol = '^GSPC'

start_date = '2019-01-01'
end_date = '2024-01-01'

stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
market_data = yf.download(market_symbol, start=start_date, end=end_date)

# drop unnecessary columns
stock_data = stock_data.drop(['Adj Close', 'Volume'], axis=1)
market_data = market_data.drop(['Adj Close', 'Volume'], axis=1)

# calculate daily returns
stock_data['Daily Return'] = stock_data['Close'].pct_change()
market_data['Market Return'] = market_data['Close'].pct_change()
stock_data['Daily Return'].iat[0] = 0
market_data['Market Return'].iat[0] = 0

# create a returns dataframe
returns = pd.concat([stock_data['Daily Return'], market_data['Market Return']], axis=1)

# calculate Beta
cov_matrix = returns.cov()
beta = cov_matrix.loc['Daily Return', 'Market Return'] / cov_matrix.loc['Market Return', 'Market Return']

print("Beta: ", beta)

# calculate expected return
risk_free_rate = 0.05
market_return = returns['Market Return'].mean() * 252  # annualized market return

# capm formula
expected_return = risk_free_rate + beta * (market_return - risk_free_rate)

print("Market return: ", market_return)
print("Expected return: ", expected_return)

# calculate Sharpe Ratio
excess_return = returns['Daily Return'] - risk_free_rate/252
sharpe_ratio = np.sqrt(252) * excess_return.mean() / excess_return.std()

print("Sharpe Ratio: ", sharpe_ratio)

# backtesting
print("------------------- Backtesting -------------------")

# calculate expected daily returns
expected_return_daily = risk_free_rate / 252 + beta * (returns['Market Return'] - risk_free_rate / 252)
returns['Expected Daily Return'] = expected_return_daily

# calculate performance metrics
mse = ((returns['Daily Return'] - returns['Expected Daily Return']) ** 2).mean()
rmse = np.sqrt(mse)
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)

# plot actual vs expected returns
plt.figure(figsize=(10,6))
plt.plot(returns.index, returns['Daily Return'], label="Actual Daily Returns", color="blue")
plt.plot(returns.index, returns['Expected Daily Return'], label="CAPM Expected Returns", color="orange")
plt.title(f"Actual vs CAPM Expected Returns for {stock_symbol}")
plt.legend()
plt.show()

# R-squared value to measure CAPM fit
from sklearn.linear_model import LinearRegression
regression_model = LinearRegression()
regression_model.fit(returns[['Expected Daily Return']].dropna(), returns['Daily Return'].dropna())
r_squared = regression_model.score(returns[['Expected Daily Return']].dropna(), returns['Daily Return'].dropna())
print("R-squared: ", r_squared)
