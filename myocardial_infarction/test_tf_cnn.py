import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import random
from keras.preprocessing.image import load_img, img_to_array

# Constants
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Dataset paths
dataset_dir = 'C:\\Users\\ashwi\\OneDrive\\Desktop\\GitHub\\Sapphire\\Sapphire\\myocardial_infarction\\dataset'
subfolders = ['train', 'val']
classes = ['infarction', 'normal']

# Load the saved model
model = keras.models.load_model(r'C:\Users\ashwi\OneDrive\Desktop\GitHub\Sapphire\Sapphire\myocardial_infarction\mi_cnn.keras')

# Initialize counters
total_images = 0
correct_predictions = 0
all_files = []

# Preprocess function
def preprocess_image(image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# First collect all files with their paths and labels
for parent in subfolders:
    for class_name in classes:
        dir_path = os.path.join(dataset_dir, parent, class_name)
        for filename in os.listdir(dir_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_files.append((os.path.join(dir_path, filename), parent, class_name))

# Shuffle all files
random.shuffle(all_files)

# Process shuffled files
print("┌──────────────────────────────────────────────────┐")
print("│               PROCESSING SHUFFLED DATA           │")
print("└──────────────────────────────────────────────────┘")

for image_path, parent, class_name in all_files:
    total_images += 1
    filename = os.path.basename(image_path)
    
    # Predict
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array, verbose=0)[0][0]
    predicted_class = "infarction" if prediction > 0.5 else "normal"
    
    # Check if correct
    is_correct = (predicted_class == class_name)
    if is_correct:
        correct_predictions += 1
    
    # Print result
    status = "✓" if is_correct else "✗"
    print(f"\n│ {status} {parent}/{class_name}/{filename}")
    print(f"│ Actual: {class_name.ljust(10)} → Predicted: {predicted_class.ljust(10)} (Confidence: {max(prediction, 1-prediction):.2%})")
    print("├" + "─" * 50)

# Calculate and print accuracy
accuracy = correct_predictions / total_images
print(f"\n┌────────────────────────────────────┐")
print(f"│          SUMMARY                  │")
print(f"├────────────────────────────────────┤")
print(f"│ Total Images Processed: {total_images}")
print(f"│ Correct Predictions: {correct_predictions}")
print(f"│ Accuracy: {accuracy:.2%}")
print(f"└────────────────────────────────────┘")