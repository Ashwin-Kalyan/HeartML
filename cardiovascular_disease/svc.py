import warnings
import time  
warnings.simplefilter("ignore", UserWarning)

# Record start time
start_time = time.time()  

import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

data = pd.read_csv("C:\\Users\\ashwi\\OneDrive\\Desktop\\GitHub\\Sapphire\\Sapphire\\cardiovascular_disease\\cardio_train_cleaned.csv")
print(data)
x = data.drop(columns='cardio', axis=1)
y = data['cardio']

scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)

x = standardized_data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# training the model
print("Training SVC model...")
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

# accuracy score for model
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)

# prediction system with actual values
with open("C:\\Users\\ashwi\\OneDrive\\Desktop\\GitHub\\Sapphire\\Sapphire\\cardiovascular_disease\\cardio_train_cleaned.csv") as f:
    reader = csv.reader(f)
    headers = next(reader)  # skip header
    input_data = [tuple(line) for line in reader]
    
    # Get the actual values (cardio column is last)
    actual_values = [int(row[-1]) for row in input_data]
    
    for i, line in enumerate(input_data[:12], 1):  # Only show first 12 patients for example
        in_data_np_arr = np.asarray(line[:-1])  # exclude the actual value
        np_arr_reshaped = in_data_np_arr.reshape(1, -1)
        std_in_data = scaler.transform(np_arr_reshaped)
        prediction = classifier.predict(std_in_data)
        
        print(f"\nPatient {i}:")
        print(f"          Prediction: {prediction}")
        print(f"          Actual: [{actual_values[i-1]}]")

# Calculate elapsed time
elapsed_time = time.time() - start_time 

# print accuracy scores and time
print("\nModel Evaluation:")
print("Accuracy on training data: ", training_data_accuracy)
print("Accuracy on test data: ", test_data_accuracy)
print(f"Time elapsed: {elapsed_time:.2f} seconds") 