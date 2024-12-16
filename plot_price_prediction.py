import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample data: Plot sizes (in square meters) and their prices (in OMR)
data = {
    'Size': [600, 900, 1000, 216, 700, 216, 216, 216, 216, 216],
    'Price': [22000, 6000, 95000, 10000, 95000, 8000, 8000, 8000, 8000, 8000]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Detect and remove outliers using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Optional log transformation to handle skewness
log_transform = False
if log_transform:
    df['Size'] = np.log(df['Size'])
    df['Price'] = np.log(df['Price'])

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

# Evaluate the model
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("Model Performance Metrics:")
print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Plot the results
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, predictions, color='red', label='Predicted Prices')
plt.xlabel('Plot Size (sqm)')
plt.ylabel('Price (OMR)')
plt.title('Plot Price Prediction in Salalah')
plt.legend()
plt.show()

# Interactive prediction
def predict_price():
    print("\nInteractive Prediction:")
    try:
        size = float(input("Enter the plot size (in square meters): "))
        if log_transform:
            size = np.log(size)
        predicted_price = model.predict([[size]])[0]
        if log_transform:
            predicted_price = np.exp(predicted_price)
        print(f"Predicted Price: {predicted_price:.2f} OMR")
    except Exception as e:
        print(f"Error: {e}")

predict_price()
