import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Iterate through diffrent dataset to check multiple stock accuracy
stock = [
    "COALINDIA.csv",
    "CIPLA.csv",
    "TITAN.csv",
    "ULTRACEMO.csv",
    "UPL.csv",
    "ADANIENT.csv",
]

# Load the data
data = pd.read_csv(stock[0])  # Update with your file path

# Convert the 'Date' column to a datetime object if it's not already
data["Date"] = pd.to_datetime(data["Date"])

# Feature engineering
data["O-C"] = data["Open"] - data["Close"]
data["H-L"] = data["High"] - data["Low"]

# Calculate the 9-period and 21-period EMAs
data["EMA_9"] = data["Close"].ewm(span=9, adjust=False).mean()
data["EMA_21"] = data["Close"].ewm(span=21, adjust=False).mean()

# Drop rows with NaN values that result from EMA calculations
data.dropna(inplace=True)

# Filter the data to only include records from one year back
end_date = data["Date"].max()  # Last date in the data
start_date = end_date - pd.DateOffset(years=1)  # One year before the last date
data = data[data["Date"] >= start_date]

# Define features (X) and target variable (y) with EMA strategy
X = data[["O-C", "H-L", "EMA_9", "EMA_21"]]
y = np.where(
    (data["Close"].shift(-1) > data["Close"]) & (data["EMA_9"] > data["EMA_21"]), 1, -1
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=44
)

# Initialize and fit the Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Now predict on test_set

y_pred = log_reg.predict(X_test)

# Investment strategy parameters
investment_amount = 10000  # Initial investment amount in dollars
position = 0  # Current position: 0 means no stock, 1 means stock is held
cash = investment_amount  # Cash available for investment
stock_value = 0  # Value of the stock held
buy_price = 0  # Price at which the stock was bought
total_value = []  # List to track the portfolio value over time

# Simulate trading based on EMA crossover and stop-loss
for i in range(len(data)):
    if data["EMA_9"].iloc[i] > data["EMA_21"].iloc[i] and position == 0:
        # Buy the stock
        position = 1
        buy_price = data["Close"].iloc[i]  # Record the buy price
        stock_value = cash / buy_price  # Number of shares bought
        cash = 0  # All cash invested
        print(f"Buy at {buy_price:.2f} on {data['Date'].iloc[i].date()}")

    elif position == 1:
        # Check for stop-loss condition
        if data["Close"].iloc[i] <= buy_price * 0.95:
            # Sell the stock if the price drops by 5%
            cash = stock_value * data["Close"].iloc[i]  # Convert stock back to cash
            stock_value = 0
            position = 0  # Position closed
            print(
                f"Stop-loss hit! Sell at {data['Close'].iloc[i]:.2f} on {data['Date'].iloc[i].date()}"
            )

        # Check for regular EMA crossover sell signal
        elif data["EMA_9"].iloc[i] < data["EMA_21"].iloc[i]:
            cash = stock_value * data["Close"].iloc[i]  # Convert stock back to cash
            stock_value = 0
            position = 0  # Position closed
            print(
                f"Sell at {data['Close'].iloc[i]:.2f} on {data['Date'].iloc[i].date()}"
            )

    # Track the total portfolio value
    if position == 1:
        total_value.append(stock_value * data["Close"].iloc[i])
    else:
        total_value.append(cash)

# Final portfolio value
final_value = cash if position == 0 else stock_value * data["Close"].iloc[-1]
print(f"Final Portfolio Value: ${final_value:.2f}")

# Plot the actual stock price and the portfolio value
plt.figure(figsize=(14, 7))
# plt.plot(data['Date'], data['Close'], label='Actual Stock Price', color='blue')
plt.plot(data["Date"], total_value, label="Portfolio Value", color="green")
plt.xlabel("Date")
plt.ylabel("Value ($)")
plt.title("Investment Strategy Using EMA Crossovers with 5% Stop-Loss (1-Year Back)")
plt.legend()
plt.xticks(rotation=45)
plt.show()
