import os
import cv2
import numpy as np
from keras.utils import to_categorical

def load_ekg_data(data_path, img_size=(128, 128), test_size=0.2):
    """
    Load EKG images from directory structure and split into train/val
    """
    categories = ['normal', 'infarction']
    X = []
    y = []
    
    for category in categories:
        path = os.path.join(data_path, 'train', category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_array = cv2.resize(img_array, img_size)
                X.append(resized_array)
                y.append(class_num)
            except Exception as e:
                print(f"Error loading {os.path.join(path, img)}: {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Preprocess
    X_train, y_train = preprocess_ekg_data(X_train, y_train)
    X_val, y_val = preprocess_ekg_data(X_val, y_val)
    
    return X_train, y_train, X_val, y_val

def preprocess_ekg_data(X, y):
    """
    Preprocess EKG data:
    - Normalize pixel values
    - Add channel dimension
    - Convert labels to one-hot encoding
    """
    # Normalize pixel values
    X = X.astype('float32') / 255.0
    
    # Add channel dimension
    X = X.reshape(-1, 1, X.shape[1], X.shape[2])
    
    # Convert labels to one-hot encoding
    y = to_categorical(y, num_classes=2)
    y = y.reshape(-1, 2, 1)
    
    return X, y