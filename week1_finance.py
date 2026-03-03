# week1_finance.py
import pandas as pd             # For reading CSV files
from sklearn.linear_model import LinearRegression  # For tiny AI model
import numpy as np              # For number calculations

# 1. Read the CSV file
df = pd.read_csv("stocks.csv")
print("Stock data:")
print(df)

# 2. Basic calculations
avg_price = df["price"].mean()
max_price = df["price"].max()
min_price = df["price"].min()

print(f"\nAverage price: {avg_price}")
print(f"Max price: {max_price}")
print(f"Min price: {min_price}")

# 3. Tiny AI model: Predict next day price for the first stock
prices = df["price"].values.reshape(-1,1)
days = np.arange(1, len(prices)+1).reshape(-1,1)
model = LinearRegression().fit(days, prices)
next_day = model.predict([[len(prices)+1]])[0][0]

print(f"\nPredicted next day price for {df['symbol'][0]}: {next_day:.2f}")

# 4. Save results to a new CSV file
df["avg_price"] = avg_price
df.to_csv("stocks_results.csv", index=False)
print("\nResults saved to stocks_results.csv")
