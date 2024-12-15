import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data: Plot sizes (in square meters) and their prices (in OMR)
data = {
    'Size': [600, 900, 1000, 216, 700, 216, 216, 216, 216, 216],
    'Price': [22000, 6000, 95000, 10000, 95000, 8000, 8000, 8000, 8000, 8000]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Split data into features (X) and target (y)
X = df[['Size']]
y = df['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict prices for the test set
predictions = model.predict(X_test)

# Plot the results
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, predictions, color='red', label='Predicted Prices')
plt.xlabel('Plot Size (sqm)')
plt.ylabel('Price (OMR)')
plt.title('Plot Price Prediction in Salalah')
plt.legend()
plt.show()
