import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import trading_indicators as ti
import numpy as np

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# create portfolio
portfolio = ['COIN', 'GRAB', 'INTC', 'MSFT', 'NVDA', 'SEAT', 'VTLE', 'GOOGL', 'IBM', 'NFLX', 'AMZN',
           'AMD', 'QQQ', 'QQQM', 'NEE', 'V', 'C', 'SQ', 'PHYS', 'SCHD', 'VOO', 'UNH', 'ABBV', 'HCA',
           'CI', 'LLY', 'ACIC', 'SBLK', 'SB', 'BP', 'VTLE', 'FSLR', 'CMG', 'JD', 'REAL', 'BABA',
           'AAP']

# RSI and Bollinger Bands Strategy
# define rsi and bollinger bands strategy
def rsi_and_bollinger_bands_strategy(df):
    ti.rsi(df)
    ti.bollinger_bands(df)
    df['Signal'] = 0
    for i in range(13, len(df)):
        if df['RSI'].iloc[i] <= 30 and df['Close'].iloc[i] <= df['Lower Band'].iloc[i]:
            df['Signal'].iloc[i] = 1
        elif df['RSI'].iloc[i] >= 70 and df['Close'].iloc[i] >= df['Upper Band'].iloc[i]:
            df['Signal'].iloc[i] = -1

# calculate returns
def calculate_rsi_bb_returns(df):
    total_profit_loss = 0
    bought = 0
    profit_loss = 0

    for i in range(13, len(df)):
        if bought > 0:
            profit_loss += (df['Gain'].iloc[i] - df['Loss'].iloc[i]) * bought

        if df['Signal'].iloc[i] == 1:
            bought += 1
            # print("Bought 1 at {}".format(round(df['Close'].iloc[i], 2)))
        elif df['Signal'].iloc[i] == -1:
            if bought > 0:
                # print("Sold {} at {}".format(bought, round(df['Close'].iloc[i], 2)))
                bought = 0
    
    total_profit_loss = profit_loss
    # print("Final price: {}".format(round(df['Close'].iloc[-1], 2)))
    return round(total_profit_loss, 2)

# calculate P/L ratio
def rsi_bb_pl_ratio(df):
    total_gain = 0
    total_loss = 0
    nwt = 0
    nlt = 0

    buy_prices = []

    for i in range(13, len(df)):
        if df['Signal'].iloc[i] == 1:
            buy_prices.append(df['Close'].iloc[i])
        elif df['Signal'].iloc[i] == -1 and len(buy_prices) > 0 or i == len(df) - 1:
            for j in range(len(buy_prices)):
                if df['Close'].iloc[i] > buy_prices[j]:
                    nwt += 1
                    total_gain += df['Close'].iloc[i] - buy_prices[j]
                else:
                    nlt += 1
                    total_loss += buy_prices[j] - df['Close'].iloc[i]
            
            buy_prices.clear()

    pl_ratio = 0
    if total_loss == 0 or nlt == 0:
        pl_ratio = total_gain / nwt
        return pl_ratio
    else:
        pl_ratio = (total_gain / nwt) / (total_loss / nlt)
        return round(pl_ratio, 2)
    
# calculate sharpe ratio
def rsi_bb_sharpe_ratio(df):
    total_rp = calculate_rsi_bb_returns(df) / 100
    rf = 0.0449
    rp = []

    df_short = df[13:]
    for i in range(14):
        beg = round(i * 35)
        end = round((i + 1) * 35)
        data = df_short[beg:end]
        rp.append(calculate_rsi_bb_returns(data))
    sigma_rp = np.std(rp)
    return round((total_rp - rf) / sigma_rp, 2)

# plot signals and closing price
def plot_rsi_bb_strategy(df):
    plt.figure(figsize=(15, 8))
    plt.plot(df['Close'], label='Close Price', alpha=0.5)
    plt.scatter(df[df['Signal'] == 1].index, df['Close'][df['Signal'] == 1], label='Buy Signal', marker='^', color='green')
    plt.scatter(df[df['Signal'] == -1].index, df['Close'][df['Signal'] == -1], label='Sell Signal', marker='v', color='red')
    plt.title('Close Price Buy and Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend(loc='best')
    plt.show()

stock_info = yf.Ticker('AMD')
stock = stock_info.history(period="2y")
rsi_and_bollinger_bands_strategy(stock)
print("Final RSI and BB profit/loss: {}".format(round(calculate_rsi_bb_returns(stock), 2)))
print("Final RSI and BB P/L ratio: {}".format(round(rsi_bb_pl_ratio(stock), 2)))
print("Final RSI and BB Sharpe ratio: {}".format(rsi_bb_sharpe_ratio(stock)))
plot_rsi_bb_strategy(stock)
