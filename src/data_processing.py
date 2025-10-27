import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def load_data(filepath):
    """ Load Stock Market Data from a CSV file"""

    data = pd.read_csv(filepath)
    return data

def preprocess_data(data, time_steps=10):
    """ Preprocess the stock market data by scaling and creating time-series sequences"""

    # select relevant features
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values

    # normalize the data
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # create sequences for the LSTM
    X, y = [], []
    for i in range(len(features_scaled) - time_steps):
        X.append(features_scaled[i:i + time_steps])
        y.append(features_scaled[i + time_steps, 3])

    return np.array(X), np.array(y), scaler


def split_data(X, y, train_size=0.8):
    """ Split data into training and testing sets"""

    train_len = int(len(X) * train_size)
    X_train, X_test = X[:train_len], X[train_len:]
    y_train, y_test = y[:train_len], y[train_len:]

    return X_train, X_test, y_train, y_test


def create_tf_dataset(X, y, batch_size):
    """ Create TensorFlow dataset for training"""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(len(X)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    filepath = "../Data/apple_stock.csv.csv"
    data = load_data(filepath)

    time_steps = 10
    X, y, scaler = preprocess_data(data, time_steps)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Data loaded and processed")
    print(f'Training samples: {len(X_train)}, Testing samples: {len(X_test)}')
