import yfinance as yf
import pandas as pd
import numpy as np

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# get data
stock_symbol = 'NVDA'
market_symbol = '^GSPC'

start_date = '2019-01-01'
end_date = '2024-01-01'

stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
market_data = yf.download(market_symbol, start=start_date, end=end_date)

# calculate daily returns
stock_data['Daily Return'] = stock_data['Adj Close'].pct_change()
market_data['Market Return'] = market_data['Adj Close'].pct_change()

returns = pd.concat([stock_data['Daily Return'], market_data['Market Return']], axis=1)
returns.columns = ['Stock Return', 'Market Return']
returns = returns.dropna()

# calculate Beta
cov_matrix = returns.cov()
beta = cov_matrix.loc['Stock Return', 'Market Return'] / cov_matrix.loc['Market Return', 'Market Return']

# calculate expected return
risk_free_rate = 0.03
market_return = returns['Market Return'].mean() * 252  # annualized market return

expected_return = risk_free_rate + beta * (market_return - risk_free_rate)

# Sharpe Ratio calculation
excess_return = returns['Stock Return'] - risk_free_rate/252
sharpe_ratio = np.sqrt(252) * excess_return.mean() / excess_return.std()

# backtesting
# CAPM strategy
initial_capital = 100000  # starting with $100,000
portfolio_value_capm = [initial_capital]

for i in range(1, len(returns)):
    daily_return = returns['Stock Return'].iloc[i] * beta
    portfolio_value_capm.append(portfolio_value_capm[-1] * (1 + daily_return))

# buy and hold strategy
portfolio_value_bh = [initial_capital]

for i in range(1, len(returns)):
    daily_return = returns['Stock Return'].iloc[i]
    portfolio_value_bh.append(portfolio_value_bh[-1] * (1 + daily_return))

portfolio_value_capm = pd.Series(portfolio_value_capm, index=returns.index)
portfolio_value_bh = pd.Series(portfolio_value_bh, index=returns.index)

# calculate max drawdown
def max_drawdown(portfolio_values):
    drawdown = 1 - portfolio_values / portfolio_values.cummax()
    return drawdown.max()

max_drawdown_capm = max_drawdown(portfolio_value_capm)
max_drawdown_bh = max_drawdown(portfolio_value_bh)

# final results
final_value_capm = portfolio_value_capm[-1]
final_value_bh = portfolio_value_bh[-1]

print(f"Calculated Beta: {beta:.2f}")
print(f"Expected Return based on CAPM: {expected_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Final Portfolio Value (CAPM-based Strategy): ${final_value_capm:.2f}")
print(f"Final Portfolio Value (Buy and Hold Strategy): ${final_value_bh:.2f}")
print(f"Max Drawdown (CAPM-based Strategy): {max_drawdown_capm:.2%}")
print(f"Max Drawdown (Buy and Hold Strategy): {max_drawdown_bh:.2%}")
