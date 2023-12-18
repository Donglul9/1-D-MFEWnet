from keras.models import Sequential
from keras.layers import *
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import Train_model
import Data_importing
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Re-create the same model to completion, including weights, optimization procedures, etc.
model = tf.keras.models.load_model(r'E:\OSR\weight_1-DMFCNN_3fenlei.h5')
# View the structure of the model
model.summary()

# Get the dataset obtained in the previous step
X_test = Data_importing.X_test
y_test = Data_importing.y_test

# Get predicted labels and true labels
prediction=model.predict(X_test)
predict_label=np.argmax(prediction,axis=1)
true_label=np.argmax(y_test,axis=1)

# Get the activation vector
def get_flist(data):
    model2 = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    features = model2(data)
    return features
f_list = get_flist(X_test)

# Non-negative activation function to compute activation vectors
def feifu(feau):
    m = np.min(feau)
    y = feau-m
    return y
f_list1 = []
for i in range(len(f_list)):
    mn = feifu(f_list[i])
    f_list1.append(mn)

# Take class i as an example, input all the training samples of class i into DCNN to get their activation vectors AV (Activation Vector)
# And retain the AVs that are correctly classified by the DCNN as samples of class i. The set of retained AVs is denoted as AVi={AV1,AV2,.... AVm}
# where m means that m samples of the training samples of class i are recognized as class i by the MFCNN.
def AV_compute(category_name,f_list):
    correct_features = []
    for i in range(len(f_list)):
        if(predict_label[i] == true_label[i]):
            if(predict_label[i] == category_name):
                correct_features.append(f_list[i])
    return correct_features
AV0 = AV_compute(0,f_list1)
AV1 = AV_compute(1,f_list1)
AV2 = AV_compute(2,f_list1)

# Use AVi to compute its mean MAVi (Mean Activation Vector), MAVi is the center of mass of the sample of class ith.
def MAV_compute(AV):
    MAV = np.mean(AV,0)
    return MAV
MAV0 = MAV_compute(AV0)
MAV1 = MAV_compute(AV1)
MAV2 = MAV_compute(AV2)
MAV = [MAV0,MAV1,MAV2]

# Using AVi={AV1,AV2,... ,AVm} in AV1,AV2,... ,AVm to compute their Euclidean distances to the center of mass MAVi
# noting that the set of distances is Di={D1, D2, ... , Dm}
def compute_distance(MAV,AV):
    query_distance =[]
    for i in range(len(AV)):
        d = np.linalg.norm(MAV-AV[i])
        query_distance.append(d)
    return query_distance
D0 = compute_distance(MAV0,AV0)
D1 = compute_distance(MAV1,AV1)
D2 = compute_distance(MAV2,AV2)

# Fit the distribution of extreme values in Di
# The distribution of great magnitudes in Di is fitted according to the Weibull classification,
# here fit_high() of libMR is used. Fitting to get Weibull
import libmr
def weibull_tailfitting(dist,tailsize):
    mr = libmr.MR()
    tailtofit = sorted(dist)[-tailsize:]
    mr.fit_high(tailtofit, tailsize)
    return mr
mr0 = weibull_tailfitting(D0, 5)
mr1 = weibull_tailfitting(D1, 5)
mr2 = weibull_tailfitting(D2, 5)

mr=[mr0,mr1,mr2]