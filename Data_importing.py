import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Read training data and split tags
dataset_ur =  r'E:\OSR\Temp-0509\chaifen\osr\train_3.csv'
data = pd.read_csv(dataset_ur)
data = np.array(data)
dataset = data[:,1:-1]
labels = data[:,-1]

# Converting tags to one-hot coding
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(labels)
lable_onehot = tf.keras.utils.to_categorical(Y_encoded)

# Constructing training and validation data
X_train, X_validation, y_train, y_validation = train_test_split(dataset,lable_onehot, test_size=0.3, random_state=20)
print("Size of training sample data：{}".format(X_train.shape))
print("Size of training sample labels：{}".format(y_train.shape))
print("Size of  validation sample data：{}".format(X_validation.shape))
print("Size of  validation labels：{}".format(y_validation.shape))

# Expanded data dimensions
X_train = np.expand_dims(X_train.astype(float), axis=2)
X_test = np.expand_dims(X_validation.astype(float), axis=2)

# Read training data and split tags
datatest_ur =  r'E:\OSR\Temp-0509\chaifen\osr\test_3.csv'
data_t = pd.read_csv(datatest_ur)
data_t = np.array(data_t)
data_test = data_t[:,1:-1]
test_labels = data_t[:,-1]

# Read test data and split tags
test_dataset_ur = r'E:\OSR\Temp-0509\chaifen\osr\test_5.csv'
test_data = pd.read_csv(test_dataset_ur)
test_data = np.array(test_data)
test_da = test_data[:,1:-1]
test_lb = test_data[:,-1]
print("Size of test sample data：{}".format(test_da.shape))
print("Size of test sample labels：{}".format(test_lb.shape))
