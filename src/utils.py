import torch
import os
import json
import matplotlib.pyplot as plt


def save_model(model, path):
    """
    Save the PyTorch model to the specified path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path):
    """
    Load a PyTorch model from the specified path.
    """
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
    else:
        raise FileNotFoundError(f"No model found at {path}")
    return model


def save_config(config, path="config/config.json"):
    """
    Save configuration settings to a JSON file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {path}")


def load_config(path="config/config.json"):
    """
    Load configuration settings from a JSON file.
    """
    if os.path.exists(path):
        with open(path, "r") as f:
            config = json.load(f)
        print(f"Configuration loaded from {path}")
        return config
    else:
        raise FileNotFoundError(f"No configuration file found at {path}")


def plot_results(actual, predicted, save_path=None):
    """
    Plot and optionally save the comparison between actual and predicted values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label="Actual Prices", color="blue")
    plt.plot(predicted, label="Predicted Prices", color="red")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Predicted vs. Actual Stock Prices")
    plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()