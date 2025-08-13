import pandas as pd 
import numpy as np
import time
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("C:\\Users\\ashwi\\OneDrive\\Desktop\\GitHub\\Sapphire\\Sapphire\\cardiovascular_disease\\cardio_train_cleaned.csv")
data = data.dropna() # Remove any rows with missing values

X = data.drop(['id', 'cardio'], axis=1) # Remove ID and target column (so our model doesn't learn from them)
y = data['cardio'] # Target variable

X = np.array(X)
y = np.array(y)

# Split into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Set random_state higher for reproducibility of results

# Standardize the features (mean=0, std=1; ensures that all features contribute equally regardless of how they are scaled)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Transpose data to match expected input shape for the FNN (features as rows, samples as columns)
X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T

# Get dimensions
n_features, m_train = X_train.shape
n_classes = 2  # Binary classification (0 or 1)

# Initialize parameters for the cardiovascular model
# Model architecture: 11 input neurons, hidden layer with 512 neurons, output layer with 2 neurons
def init_params():
    W1 = np.random.randn(512, n_features) * np.sqrt(2.0/(n_features+512))  # Xavier initialization
    B1 = np.zeros((512, 1))
    W2 = np.random.randn(n_classes, 512) * np.sqrt(2.0/(512+n_classes))
    B2 = np.zeros((n_classes, 1))
    return W1, B1, W2, B2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z))
    return exp_Z / exp_Z.sum(axis=0)

def forward_propagation(W1, B1, W2, B2, X):
    Z1 = W1 @ X + B1 # @ is matrix multiplication
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + B2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, n_classes))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def derivative_ReLU(Z):
    return Z > 0

def back_propagation(Z1, A1, Z2, A2, W1, W2, X, Y, lamda_reg):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (dZ2 @ A1.T) / m + ((lamda_reg / m) * W2) # L2 Regularization
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = W2.T @ dZ2 * derivative_ReLU(Z1)
    dW1 = (dZ1 @ X.T) / m + ((lamda_reg / m) * W1)
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return dW1, db1, dW2, db2

def update_params(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha):
    W1 -= alpha * dW1
    B1 -= alpha * dB1
    W2 -= alpha * dW2
    B2 -= alpha * dB2
    return W1, B1, W2, B2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

# Function to generate batches for mini-batch gradient descent
# This allows us to process smaller chunks of data at a time, which can help with convergence
def batch_generator(X, Y, batch_size):
    m = X.shape[1]
    for i in range(0, m, batch_size):
        yield X[:, i:i+batch_size], Y[i:i+batch_size]

def gradient_descent(X, Y, X_test, Y_test, iterations, batch_size, initial_alpha):
    W1, b1, W2, b2 = init_params()
    m = X.shape[1]
    start_time = time.time()
    
    # For tracking performance
    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []
    
    # Early stopping variables
    best_test_acc = 0
    no_improve = 0
    patience = 10
    min_improvement = 0.0001
    
    for i in range(iterations):
        # Shuffle data at epoch boundaries
        if i % (m // batch_size) == 0:
            perm = np.random.permutation(m)
            X_shuffled = X[:, perm]
            Y_shuffled = Y[perm]
        
        # Learning rate decay
        n_min = 0.001
        t = i
        T = iterations
        alpha = n_min + 0.5 * (initial_alpha - n_min) * (1 + math.cos((math.pi * t) / T)) # Cosine annealing
        
        # Process batches
        for X_batch, Y_batch in batch_generator(X_shuffled, Y_shuffled, batch_size):
            Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X_batch)
            dW1, db1, dW2, db2 = back_propagation(Z1, A1, Z2, A2, W1, W2, X_batch, Y_batch, 0.01)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        # Progress monitoring
        if i % 20 == 0:
            # Training metrics
            train_pred = get_predictions(A2)
            train_acc = get_accuracy(train_pred, Y_batch)
            train_loss = -np.mean(np.log(A2[Y_batch, np.arange(Y_batch.size)])) # Cross-entropy loss
            
            # Test metrics
            _, _, _, A2_test = forward_propagation(W1, b1, W2, b2, X_test)
            test_pred = get_predictions(A2_test)
            test_acc = get_accuracy(test_pred, Y_test)
            test_loss = -np.mean(np.log(A2_test[Y_test, np.arange(Y_test.size)]))
            
            # Store metrics
            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)
            train_acc_history.append(train_acc)
            test_acc_history.append(test_acc)
            
            elapsed = time.time() - start_time
            remaining = (iterations-i) * (elapsed/(i+1))
            
            print(f"Iter {i}: {elapsed:.1f}s | ~{remaining:.1f}s remaining | "
                  f"Train Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f} | "
                  f"Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f} | LR: {alpha:.5f}")
            
            # Early stopping check
            if test_acc > best_test_acc + min_improvement:
                best_test_acc = test_acc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\nEarly stopping at iteration {i}")
                    print(f"Best test accuracy: {best_test_acc:.4f}\n")
                    return W1, b1, W2, b2, train_loss_history, test_loss_history, train_acc_history, test_acc_history
    
    return W1, b1, W2, b2, train_loss_history, test_loss_history, train_acc_history, test_acc_history

def evaluate_model(X, Y, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    accuracy = get_accuracy(predictions, Y)
    
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(Y, predictions))
    
    # Confusion matrix
    cm = confusion_matrix(Y, predictions)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No CVD', 'CVD'], 
                yticklabels=['No CVD', 'CVD'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def analyze_feature_importance(W1, W2, feature_names):
    # Calculate feature importance by multiplying input-to-hidden weights
    # with hidden-to-output weights (absolute values)
    importance = np.sum(np.abs(W1), axis=0)  # Sum across hidden units (shape: n_features)
    
    # Normalize
    importance = importance / np.sum(importance)
    
    # Create a DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

# Start training with optimized parameters
print("Starting training...")
W1, B1, W2, B2, train_loss, test_loss, train_acc, test_acc = gradient_descent(
    X_train, y_train, X_test, y_test, 500, 64, 0.05)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.title('Loss over iterations')
plt.xlabel('Iterations (x20)')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Train Accuracy')
plt.plot(test_acc, label='Test Accuracy')
plt.title('Accuracy over iterations')
plt.xlabel('Iterations (x20)')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluate on test set
print("\nEvaluating on test set:")
evaluate_model(X_test, y_test, W1, B1, W2, B2)

# Feature importance analysis
feature_names = data.drop(['id', 'cardio'], axis=1).columns.tolist()
analyze_feature_importance(W1, W2, feature_names)

# Function to display predictions vs actual for sample patients
def show_sample_predictions(X, Y, W1, b1, W2, b2, num_samples=100):
    # Get predictions for all test samples
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    
    # Select random samples (or first num_samples)
    sample_indices = np.random.choice(range(X.shape[1]), size=num_samples, replace=False)
    
    print("\nSample Patient Predictions vs Actual:")
    print("------------------------------------")
    
    correct = 0
    incorrect = 0
    
    for i, idx in enumerate(sample_indices, 1):
        patient_data = X[:, idx]
        prediction = predictions[idx]
        actual = Y[idx]
        
        # Inverse transform the standardized data for display
        patient_data_original = scaler.inverse_transform(patient_data.reshape(1, -1))[0]
        
        print(f"\nPatient {i}:")
        print(f"Features: {dict(zip(feature_names, patient_data_original))}")
        print(f"Prediction: {prediction} ({'Has CVD' if prediction == 1 else 'No CVD'})")
        print(f"Actual: {actual} ({'Has CVD' if actual == 1 else 'No CVD'})")
        
        if prediction == actual:
            correct += 1
            print("Result: CORRECT")
        else:
            incorrect += 1
            print("Result: INCORRECT")
    
    # Calculate and display statistics
    accuracy = correct / (correct + incorrect) * 100
    print("\n\nSummary Statistics:")
    print("------------------")
    print(f"Total Patients: {num_samples}")
    print(f"Correct Predictions: {correct}")
    print(f"Incorrect Predictions: {incorrect}")
    print(f"Accuracy Rate: {accuracy:.2f}%")
    
    return correct, incorrect, accuracy

# Show predictions for sample patients
correct, incorrect, sample_accuracy = show_sample_predictions(X_test, y_test, W1, B1, W2, B2, num_samples=100)