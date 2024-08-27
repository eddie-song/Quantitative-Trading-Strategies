import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# get data as csvs
portfolio = ['COIN', 'GRAB', 'INTC', 'MSFT', 'NVDA', 'SEAT', 'VTLE', 'GOOGL', 'IBM', 'NFLX', 'AMZN',
           'AMD', 'QQQ', 'QQQM', 'NEE', 'V', 'C', 'SQ', 'PHYS', 'SCHD', 'VOO', 'UNH', 'ABBV', 'HCA',
           'CI', 'LLY', 'ACIC', 'SBLK', 'SB', 'BP', 'VTLE', 'FSLR', 'CMG', 'SBUX', 'JD', 'REAL', 'BABA', 'AAP']
df = yf.download(portfolio, start='2015-01-01', end=datetime.today().strftime('%Y-%m-%d'))['Adj Close']

assets = portfolio
weights = np.array([0.0477, 0.018, 0.239, 0.0986, 0.1017, 0.0071, 0.0274, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

title = 'Portfolio Adj. Close Price History'

# plot the adjusted close price history
for c in df.columns.values:
    plt.plot(df[c], label=c)

plt.title(title)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Adj. Price USD ($)', fontsize=18)
plt.legend(df.columns.values, loc='upper left')
plt.show()

# calculate daily returns
returns = df.pct_change()

# annualized covariance matrix
cov_matrix_annual = returns.cov() * 252

# calculate portfolio variance
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))

# calculate portfolio volatility or standard deviation
port_volatility = np.sqrt(port_variance)

# calculate annual portfolio returns
portfolioSimpleAnnualReturns = np.sum(returns.mean() * weights) * 252

# show the expected annual returns, volatility (risk), variance for the initial portfolio
percent_var = str(round(port_variance, 4) * 100) + '%'
percent_vols = str(round(port_volatility, 4) * 100) + '%'
percent_ret = str(round(portfolioSimpleAnnualReturns, 4) * 100) + '%'
print('Initial Portfolio:')
print('Expected Annual Return: ' + percent_ret)
print('Annual Volatility/Risk: ' + percent_vols)
print('Annual Variance: ' + percent_var)

# optimization
# calculate the expected returns and the annualized simple covariance matrix of assets
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

ef = EfficientFrontier(mu, S)

# set weight constraints
ef.add_objective(objective_functions.L2_reg, gamma=0.1)  # adding a slight L2 regularization for more diversification
ef.add_constraint(lambda x: x >= 0.0)
ef.add_constraint(lambda x: x <= 0.10)

# optimize for max Sharpe ratio within constraints
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

# display the results
optimized_ret, optimized_vol, _ = ef.portfolio_performance(verbose=True)

# calculate the increase in expected return and volatility
increase_in_return = optimized_ret - portfolioSimpleAnnualReturns
increase_in_volatility = optimized_vol - port_volatility

print('\nIncrease in Expected Annual Return: {:.2f}%'.format(increase_in_return * 100))
print('Increase in Annual Volatility/Risk: {:.2f}%'.format(increase_in_volatility * 100))

# get the discrete allocation of each share per stock
latest_prices = get_latest_prices(df)
da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=4100)

allocation, leftover = da.lp_portfolio()
print('Discrete allocation:', allocation)
print('Funds remaining: ${:.2f}'.format(leftover))
