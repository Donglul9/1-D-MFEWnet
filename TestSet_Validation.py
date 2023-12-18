from keras.models import Sequential
from keras.layers import *
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import Data_importing
import Train_model
import Fit_weibull
import TrainSet_Validation
import pandas as pd
import Classifier
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set the appropriate parameters
NCLASSES = 3  # Number of known events
disatances = []
ranked_alpha=[1.0,0.667,0.333]
thr=0.8

# Get the parameters obtained in the previous step
MAV = Fit_weibull.MAV
mr = Fit_weibull.mr

# Re-create the same model to completion, including weights, optimization procedures, etc.
model = tf.keras.models.load_model(r'E:\OSR\weight_mcnn.h5')
# View the structure of the model
model.summary()

# Read test dataset
test_da = Data_importing.test_da
test_lb = Data_importing.test_lb

# Obtain AV, the output of the penultimate layer of the model
model2 = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
features = model2(test_da)
features

# Applying a non-negative activation function to AV
m_score = []
m_score = Classifier.feifu(features)

# Applying a non-negative activation function to the AV to obtain the distance of the test data from the MAV
for categoryid in range(NCLASSES):
    D0 = Classifier.compute_distance(MAV[categoryid], m_score)
    disatances.append(D0)

# Modify the AV to get the modified score
Score_final = []
for i in range(len(test_da)):
    Score_final = Classifier.modify_score(test_da[i])
# Get the corresponding probability
fs = []
for i in range(len(Score_final)):
    p = Classifier.softmax(Score_final[i])
    fs.append(p)
# Threshold judgment
fp = []
for i in range(len(fs)):
    pp = Classifier.th(fs[i])
    fp.append(pp)
fp = np.array(fp)

# Constructing confusion matrices
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=test_lb, y_pred=fp)
attack_types = [ 'Environment', 'Walk','Car','Unknown']
plt.figure(figsize=(12,4),dpi=100)
TrainSet_Validation.plot_confusion_matrix(cm,classes=attack_types, normalize=False, title='Predict Value')

# Evaluation criteria
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
print('Precision: %.3f' % precision_score(test_lb, fp,average='macro'))
print('Recall: %.3f' % recall_score(test_lb, fp,average='macro'))
print('Accuracy: %.3f' % accuracy_score(test_lb, fp))
print('F1 Score: %.3f' % f1_score(test_lb, fp,average='macro'))


