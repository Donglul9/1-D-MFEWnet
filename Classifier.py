from keras.models import Sequential
from keras.layers import *
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import Fit_weibull
import pandas as pd

# Set the appropriate parameters
NCLASSES = 3  # Number of known events
disatances = []
ranked_alpha=[1.0,0.667,0.333]
thr=0.8

# Get the parameters obtained in the previous step
MAV = Fit_weibull.MAV
mr = Fit_weibull.mr

# Re-create the same model to completion, including weights, optimization procedures, etc.
model = tf.keras.models.load_model(r'E:\OSR\weight_1-DMFCNN_3fenlei.h5')
# View the structure of the model
model.summary()

# Test data read
test_dataset_ur = r'E:\OSR\Temp-0509\chaifen\osr\test.csv'
test_da = pd.read_csv(test_dataset_ur)


# Obtain AV, the output of the penultimate layer of the model
model2 = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
features = model2(test_da)
features

# Applying a non-negative activation function to AV
m_score = []
def feifu(feau):
    m = np.min(feau)
    y = feau-m
    return y
m_score =feifu(features)

# Applying a non-negative activation function to the AV to obtain the distance of the test data from the MAV
def compute_distance(MAV,AV):
    query_distance =[]
    for i in range(len(AV)):
        d = np.linalg.norm(MAV-AV[i])
        query_distance.append(d)
    return query_distance
for categoryid in range(NCLASSES):
    D0 = compute_distance(MAV[categoryid], m_score)
    disatances.append(D0)

# Modify the AV to get the modified score
def modify_score(test_f):
    wscore = []
    modified_fc_score = []
    fc_unknown = 0
    for categoryid in range(NCLASSES):
        category_weibull = mr[categoryid]
        ws = category_weibull.w_score(disatances[categoryid])
        wscore.append(ws)
    for categoryid in range(NCLASSES):
        if (wscore[categoryid] == max(wscore)):
            modified_score = test_f[categoryid] * (1 - wscore[categoryid] * ranked_alpha[0])
        elif (wscore[categoryid] == min(wscore)):
            modified_score = test_f[categoryid] * (1 - wscore[categoryid] * ranked_alpha[2])
        else:
            modified_score = test_f[categoryid] * (1 - wscore[categoryid] * ranked_alpha[10])
        modified_fc_score += [modified_score]
        fc_unknown += test_f[categoryid]-modified_score
    Score_final = modified_fc_score+[fc_unknown]
    Score_final = np.array(Score_final)
    return Score_final
Score_final = modify_score(m_score)

# Define the softmax function and get the corresponding probability
f_probab = []
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
f_probab = softmax(Score_final)

# Threshold judgment, if the maximum probability is less than the threshold, it is judged as an unknown event category
def th(score):
    m = max(score)
    if(m<thr):
        p = 3
    else:
        p = np.argmax(score)
    return p

# Get the type of test data
events = ['Environment','Walk','Car','Unknown']
f_p = th(f_probab)
print("This event is %s"%(events[f_p]))












