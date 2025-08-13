import cv2
import numpy as np

def preprocess_ekg_image(image):
    """
    Apply EKG-specific preprocessing to a single image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to remove background
    _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find and crop to EKG trace
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2*padding)
        h = min(image.shape[0] - y, h + 2*padding)
        image = image[y:y+h, x:x+w]
    
    # Resize and enhance contrast
    image = cv2.resize(image, (128, 128))
    image = cv2.equalizeHist(image)
    
    return image

def augment_ekg_image(image):
    """
    Apply random augmentations to EKG image
    """
    # Random rotation (-5 to +5 degrees)
    angle = np.random.uniform(-5, 5)
    rows, cols = image.shape[1], image.shape[2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    image = cv2.warpAffine(image[0], M, (cols, rows))
    image = image.reshape(1, rows, cols)
    
    # Random brightness adjustment
    brightness = np.random.uniform(0.9, 1.1)
    image = np.clip(image * brightness, 0, 255)
    
    return image