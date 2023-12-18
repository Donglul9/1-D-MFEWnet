import itertools
from keras.models import Sequential
from keras.layers import *
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import Train_model
import Data_importing
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Re-create the same model to completion, including weights, optimization procedures, etc.
model = tf.keras.models.load_model(r'E:\OSR\weight_mcnn.h5')
# View the structure of the model
model.summary()

# Read dataset
data_test = Data_importing.data_test
test_labels = Data_importing.test_labels

# T-SNE
from sklearn.manifold import TSNE
model2 = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
X = model2(data_test)
Y = np.argmax(test_labels,axis=1)
tsne = TSNE(n_components=2, perplexity=15).fit_transform(X)
aa = tsne[:, 0]
bb = tsne[:, 1]

color = ['limegreen', 'cornflowerblue', 'orange']

plt.figure(dpi=100)
for i in range(tsne.shape[0]):
    plt.scatter(aa[i], bb[i], facecolor=color[Y[i]], alpha=0.7)

# Constructing confusion matrices
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    - cm : The value of the computed confusion matrix
    - classes : Columns corresponding to each row and each column in the confusion matrix
    - normalize : True:Show percentage, False:Show number of items
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Show Percentage：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('Show specific numbers：')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Test data softmax classification results
prediction=model.predict(data_test)
predict_label=np.argmax(prediction,axis=1)
true_label=np.argmax(test_labels,axis=1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=true_label, y_pred=predict_label)
attack_types = [ 'Environment', 'Walk','Car']
plt.figure(figsize=(12,4),dpi=100)
plot_confusion_matrix(cm,classes=attack_types, normalize=False, title='Predict Value')

# Evaluation criteria
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
print('Precision: %.3f' % precision_score(true_label, predict_label,average='macro'))
print('Recall: %.3f' % recall_score(true_label, predict_label,average='macro'))
print('Accuracy: %.3f' % accuracy_score(true_label, predict_label))
print('F1 Score: %.3f' % f1_score(true_label, predict_label,average='macro'))
