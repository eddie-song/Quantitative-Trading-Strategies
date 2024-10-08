import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score

# load data
ticker = 'NVDA'  # Example ticker
nvda = yf.Ticker(ticker)
data = nvda.history(start='2020-01-01', end='2024-08-13')  # Ensures the most up-to-date data
data = data[['Close']]
data['Prediction'] = data['Close'].shift(-30)  # Predict 30 days into the future
data = data.dropna()

# feature engineering
data['Moving_Avg_30'] = data['Close'].rolling(window=30).mean()
data['Moving_Avg_50'] = data['Close'].rolling(window=50).mean()
data = data.dropna()

# train data
X = data[['Close', 'Moving_Avg_30', 'Moving_Avg_50']]
y = data['Prediction']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train svr model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(X_train, y_train)

# make predictions
y_pred = svr_rbf.predict(X_test)

# plot results
plt.figure(figsize=(14, 7))
plt.plot(data.index[-len(y_test):], y_test.values, label='Actual Prices')
plt.plot(data.index[-len(y_test):], y_pred, label='Predicted Prices', color='red')
plt.title('SVR Model Prediction vs Actual')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()

# back test model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# retrain the model
X_full = data[['Close', 'Moving_Avg_30', 'Moving_Avg_50']]
y_full = data['Prediction']

X_full_scaled = scaler.fit_transform(X_full)

svr_rbf.fit(X_full_scaled, y_full)

# future predictions
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')  # B for business days

future_data = pd.DataFrame(index=future_dates, columns=data.columns[:-1])

future_data['Close'] = data['Close'].iloc[-1] * (1 + np.linspace(0, 0.02, 30))  # A 2% increase over 30 days

future_data['Moving_Avg_30'] = future_data['Close'].rolling(window=30, min_periods=1).mean()
future_data['Moving_Avg_50'] = future_data['Close'].rolling(window=50, min_periods=1).mean()

future_data = future_data[['Close', 'Moving_Avg_30', 'Moving_Avg_50']]  # Ensuring column order

X_future_scaled = scaler.transform(future_data)

future_predictions = svr_rbf.predict(X_future_scaled)

# append predictions
future_data['Predicted_Close'] = future_predictions

# plot results
plt.figure(figsize=(14, 7))

# plot historical prices
plt.plot(data.index, data['Close'], label='Historical Prices')

# plot predictions
plt.plot(future_data.index, future_data['Predicted_Close'], label='Predicted Prices', color='red')

plt.title('SVR Model Future Predictions')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# print final prediction
print(future_data[['Predicted_Close']])
