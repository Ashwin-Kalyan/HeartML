import numpy as np
from model.dense import Dense
from model.activations import Sigmoid, Tanh
from model.losses import binary_cross_entropy, binary_cross_entropy_prime
from model.convolution import Convolutional
from model.reshape import Reshape
from network_actions import train, predict, evaluate
from data_loader import load_ekg_data

def main():
    # Load and preprocess data
    data_path = 'C:\\Users\\ashwi\\OneDrive\\Desktop\\GitHub\\Sapphire\\Sapphire\\myocardial_infarction\\dataset'  # Update with your path
    X_train, y_train, X_val, y_val = load_ekg_data(data_path)

    # Define network architecture
    network = [
        Convolutional((1, 128, 128), 5, 16),  # Input: 1x128x128 (grayscale)
        Tanh(),
        Convolutional((16, 124, 124), 3, 32),  # Second conv layer
        Tanh(),
        Reshape((32, 122, 122), (32 * 122 * 122, 1)),
        Dense(32 * 122 * 122, 512),
        Tanh(),
        Dense(512, 128),
        Tanh(),
        Dense(128, 2),
        Sigmoid()
    ]

    # Train the network
    train(
        network,
        binary_cross_entropy,
        binary_cross_entropy_prime,
        X_train,
        y_train,
        epochs=50,
        learning_rate=0.001
    )

    # Evaluate on validation set
    evaluate(network, X_val, y_val)

if __name__ == "__main__":
    main()