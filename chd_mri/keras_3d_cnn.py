import os
import numpy as np
import nibabel as nib
from keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data(data_dir, target_shape=(128, 128, 64)):
    """Load and preprocess all .nii.gz files with shape validation"""
    volumes = []
    nii_files = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
    
    if not nii_files:
        raise ValueError(f"No .nii.gz files found in {data_dir}")

    for fname in nii_files:
        try:
            # Load volume
            img = nib.load(os.path.join(data_dir, fname))
            data = img.get_fdata()
            
            # Skip if loading failed
            if data is None:
                print(f"Warning: Could not load data from {fname}")
                continue
                
            # Normalize volume
            data_min, data_max = np.min(data), np.max(data)
            if data_max > data_min:  # Avoid division by zero
                data = (data - data_min) / (data_max - data_min)
            else:
                data = np.zeros_like(data)
            
            # Validate shape before processing
            if any(d < t for d, t in zip(data.shape, target_shape)):
                print(f"Skipping {fname}: Input shape {data.shape} < target {target_shape}")
                continue
                
            # Center crop to target shape
            crop = [(d - t) // 2 for d, t in zip(data.shape, target_shape)]
            cropped = data[
                crop[0]:crop[0]+target_shape[0],
                crop[1]:crop[1]+target_shape[1],
                crop[2]:crop[2]+target_shape[2]
            ]
            
            # Verify final shape
            if cropped.shape != target_shape:
                print(f"Skipping {fname}: Cropped shape {cropped.shape} != target {target_shape}")
                continue
                
            volumes.append(cropped)
            
        except Exception as e:
            print(f"Error processing {fname}: {str(e)}")
            continue
    
    if not volumes:
        raise ValueError("No valid volumes were loaded")
    
    # Convert to numpy array with explicit shape check
    try:
        volume_array = np.stack(volumes)  # Instead of np.array()
        return np.expand_dims(volume_array, -1)  # Add channel dimension
    except ValueError as e:
        raise ValueError(f"Volume shape mismatch: {e}. All volumes must be {target_shape}") from e

def build_3d_autoencoder(input_shape=(128, 128, 64, 1)):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling3D((2, 2, 2))(x)
    
    # Decoder
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling3D((2, 2, 2))(x)
    
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling3D((2, 2, 2))(x)
    
    x = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling3D((2, 2, 2))(x)
    
    decoded = layers.Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = models.Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder(data_dir):
    # Load data
    volumes = load_data(data_dir)
    print(f"Loaded {len(volumes)} volumes with shape {volumes[0].shape}")
    
    # Split data (we'll use all for training since we have no negatives)
    train_vol, val_vol = train_test_split(volumes, test_size=0.2, random_state=42)
    
    # Build model
    autoencoder = build_3d_autoencoder(input_shape=volumes.shape[1:])
    autoencoder.summary()
    
    # Train
    history = autoencoder.fit(train_vol, train_vol,
                            epochs=50,
                            batch_size=4,
                            validation_data=(val_vol, val_vol))
    
    # Plot training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()
    
    return autoencoder

def detect_anomalies(autoencoder, volume, threshold=0.1):
    """Returns whether the scan appears healthy (high MSE) or CHD (low MSE)"""
    reconstruction = autoencoder.predict(np.expand_dims(volume, axis=0))
    mse = np.mean(np.square(volume - reconstruction[0]))

    is_healthy = mse > threshold  # High error = unfamiliar (healthy) pattern
    return is_healthy, mse

# Usage Example
if __name__ == "__main__":
    data_dir = "C:\\Users\\ashwi\\OneDrive\\Desktop\\GitHub\\Sapphire\\Sapphire\\chd_mri\\data"
    autoencoder = train_autoencoder(data_dir)
    
    # Save model
    autoencoder.save("chd_autoencoder.keras")
    
    # Example detection on a new volume
    sample_volume = load_data(data_dir)[0]  # Just using first volume as example
    is_anomalous, score = detect_anomalies(autoencoder, sample_volume)
    print(f"Anomaly detected: {is_anomalous} (score: {score:.4f})")