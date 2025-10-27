import tensorflow as tf
import json
import os
from src.data_processing import load_data, preprocess_data, create_tf_dataset
from src.model import build_lstm_model

# load config
with open("../config/config.json") as f:
    config = json.load(f)

# load and preprocess data
data = load_data(config["dataset_path"])
X, y, scaler = preprocess_data(data, config["time_steps"])

# split data into training and testing
split_idx = int((1 - config["test_size"]) * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# create datasets
train_dataset = create_tf_dataset(X_train, y_train, batch_size=config['batch_size'])

# build model
model = build_lstm_model(time_steps=config["time_steps"], num_features=5,
                         hidden_size=config["hidden_size"], num_layers=config["num_layers"],
                         dropout=config["dropout"])

# learning rate scheduler
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * 0.9 ** (epoch // 10))

# train model
model.fit(train_dataset, epochs=config["epochs"], callbacks=[lr_scheduler])

# Ensure directory exists
os.makedirs("../outputs/models", exist_ok=True)

# Save model
model.save("../outputs/models/stock_lstm_tf.keras")
print("Model saved to outputs/models/stock_lstm_tf.keras")

