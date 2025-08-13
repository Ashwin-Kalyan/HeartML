import numpy as np
from keras.utils import to_categorical
from ekg_preprocessing import augment_ekg_image
import time

def train(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=True):
    samples = len(x_train)
    for e in range(epochs):
        error = 0
        correct = 0
        start_time = time.time()
        
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            # Data augmentation
            if np.random.rand() > 0.5:
                x = augment_ekg_image(x)
                
            # Forward pass
            output = predict(network, x)
            
            # Calculate metrics
            error += loss(y, output)
            correct += int(np.argmax(output) == np.argmax(y))
            
            # Backward pass
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
            
            # Print batch progress
            if verbose and i % 10 == 0:
                batch_acc = correct/(i+1)
                print(f"\rEpoch {e+1}/{epochs} | Batch {i+1}/{samples} | "
                      f"Batch Acc: {batch_acc:.2%} | "
                      f"Avg Loss: {error/(i+1):.4f}", end='', flush=True)
        
        # Calculate epoch metrics
        epoch_loss = error/samples
        epoch_acc = correct/samples
        epoch_time = time.time() - start_time
        
        if verbose:
            print(f"\rEpoch {e+1}/{epochs} | Time: {epoch_time:.1f}s | "
                  f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2%}        ")
    
    return epoch_loss, epoch_acc

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def evaluate(network, x_val, y_val):
    correct = 0
    total = len(x_val)
    for x, y in zip(x_val, y_val):
        output = predict(network, x)
        predicted_class = np.argmax(output)
        true_class = np.argmax(y)
        if predicted_class == true_class:
            correct += 1
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.2%}")
    return accuracy