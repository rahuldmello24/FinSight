import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
from src.data_processing import load_data, preprocess_data
from src.model import build_lstm_model

# load config
with open("../config/config.json") as f:
    config = json.load(f)

# Load and preprocess data
data = load_data(config["dataset_path"])
X, y, scaler = preprocess_data(data, config["time_steps"])

# Split data into training and testing
split_idx = int((1 - config["test_size"]) * len(X))
X_test = X[split_idx:]
y_test = y[split_idx:]

# load trained model
model = tf.keras.models.load_model("../outputs/models/stock_lstm_tf.keras")
print('Model loaded for evaluation')

# make predictions
predictions = model.predict(X_test).flatten()

# inverse transform predictions
# Reconstruct predictions with correct shape for inverse transformation
reconstructed_predictions = np.zeros((predictions.shape[0], scaler.n_features_in_))  # Create an empty array with correct shape
reconstructed_predictions[:, 3] = predictions  # Place predictions in the correct column (Close Price)

# Apply inverse transform
predictions = scaler.inverse_transform(reconstructed_predictions)[:, 3]

# Reconstruct y_test with correct shape for inverse transformation
reconstructed_y_test = np.zeros((y_test.shape[0], scaler.n_features_in_))  # Create empty array with correct shape
reconstructed_y_test[:, 3] = y_test  # Place y_test in 'Close' column

# Apply inverse transform
y_test = scaler.inverse_transform(reconstructed_y_test)[:, 3]

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions[:len(y_test)] - y_test) ** 2))

normalized_rmse = rmse / np.mean(y_test)
print(f"RMSE on Test Data: {rmse:.4f}")
print(f"Normalized RMSE: {normalized_rmse:.4f}")

# Plot predictions vs. actual prices
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Actual Prices", color="blue")
plt.plot(predictions, label="Predicted Prices", color="red")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Predicted vs. Actual Stock Prices")
plt.legend()
plt.savefig("../outputs/plots/predicted_vs_actual_tf.png")
plt.show()

