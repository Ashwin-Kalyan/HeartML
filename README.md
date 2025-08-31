# Cardiovascular Disease Predictor

Data Source: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
Data Info: 70,000 patients, 13 data points

What is cardiovascular disease? 

Conditions affecting the heart and blood vessels. Encompasses a range of disorders including 
coronary artery disease, stroke, heart failure, and preipheral artery disease. Atherosclerosis 
is the common underlying cause of these conditions. 

I use two distinct models to predict cardiovascular disease. 

The first is a feedforward neural network. This is built from scratch.

This network has an architecture consisting of 11 input neurons, 512 hidden neurons, and 2 output neurons.
The inputs are age (days), gender (0 = M; 1 = F), height (cm), weight (kg), systolic blood pressure (mmHg)--
the top number in a BP reading; represents the pressure in your arties when your heart pumps blood out; 
shown in data as "ap_hi", diastolic blood pressure (mmHg)--the bottom number in BP reading; represents the 
minimum arterial pressure between heartbeats when the ventricles are relaxed; shown as "ap_lo", glocuse levels
(1: normal, 2: above avg; 3: well above avg), smoking (0 = F; 1 = T), alcohol (0 = F; 1 = T), & activity (0 = F; 1 = T).
It uses ReLU and Softmax for its activation functions.

Libraries Used: 
    Scikit-Learn, 
    MatPlotLib, 
    Seaborn, 
    Pandas, 
    Numpy, 
    Time, 
    Math
Features: 
    Xavier Initialization, 
    L2 Regularization, 
    Feature Standardization, 
    Cosine Annealing for Learning Rate Decay,
    Mini-Batch Processing
    Early Stop
    Graphing loss, accuracy, confusion matrix, feature importance overtime
    Predicts for 100 patients
Stats:
    Iterations: ~480
    Train Loss: 0.557
    Test Loss: 0.543
    Train Acc: 0.719
    Test Acc: 0.742
    Learning R: 0.05 -> 0.001
    Patient Acc: ~61-81%
    F1 Score: 0.74
    Training Time: 113.6s

The second in a Support Vector Classifier. 

We code this using Scikit-Learn's pre-built libraries. Same inputs as the FNN. 

Libraries Used:
    Scikit-Learn.
    Pandas,
    Numpy,
    Time,
    CSV
Features:
    Standardization,
    Linear Kernel,
    Predicts for 12 patients
Stats:
    Train Acc: 0.724
    Test Acc: 0.721
    Training Time: 145.8s

# Myocardial Infarction

Data Source: https://www.kaggle.com/datasets/jayaprakashpondy/ecgimages
Data Info: 1,376 Images

What is myocardial infarction?

Myocardial infarction (MI), commonly known as a heart attack, is a serious medical condition where blood flow to the heart muscle is abruptly blocked, depriving it of oxygen and nutrients. This blockage can cause damage or death to the heart muscle. 

I used two methods for creating the CNN that examines EKG data to determine whether a patient has myocardial infarction or not.

The first is Tensorflow Keras. This uses ReLU and Sigmoid as its activation functions.

Libraries Used:
    Tensorflow Keras
    MatPlotLib
    OS
Features:
    4 Convolutional Blocks
    Dropout (.2 -> .5)
    32 5x5 kernels
    Normalization
    Graphs accuracy and loss over time
    Predicts for all 1376 patients
Stats:
    Epochs: 75
    Train Acc: 0.8473
    Train Loss: 0.3576
    Train Precision: 0.8341
    Train Recall: 0.8304
    Train AUC: 0.9211
    Test Acc: 0.9464
    Test Loss: 0.2280
    Test Precision: 1.0
    Test Recall: 0.8929
    Test AUC: 0.9944
    Train Time: ~ 1.5 hrs
    Patient Acc: 0.9518

The second is a convolutional neural network written from scratch.

# Conginetal Heart Defect

Data: https://figshare.com/articles/dataset/HVSMR-2_0_orig_/25226360?backTo=/collections/HVSMR-2_0_A_3D_cardiovascular_MR_dataset_for_whole-heart_segmentation_in_congenital_heart_disease/7074755

So I was not able to find a data source that included both normal and positive cases of CHD, so I reframed this project as an anamoly detection (one-class classification) problem. I could've hypothetically gotten a control case of normal MRI images but that might've just confused the model especially if they were formatted differently or of different areas of the heart. 

I use Keras to create a Cardiac MRI Autoencoder using a 3D convolutional neural network. The model will essentially compress and reconstruct the CHD MRI images. It will be able to do this very well with a low Mean-Squared Error, since the only data I gave it are positive cases. If it is given a normal scan it will have a harder time reconstructing it since it will learn the abnormalities and specific patterns only seen in CHD cases, thus resulting in a higher MSE for reconstructing normal cases. Thus, if the MSE passes a certain threshold, we will know that it is a normal patient, and if it is below the threshold it is a CHD patient.  