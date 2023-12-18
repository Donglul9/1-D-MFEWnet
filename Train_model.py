from keras.models import Sequential
from keras.layers import *
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, LSTM, Dense
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
import matplotlib.pyplot as plt
import Data_importing
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Read data and split tags
X_train = Data_importing.X_train
X_validation = Data_importing.X_test
y_train =Data_importing.y_train
y_validation = Data_importing.y_test

# Network Model Construction
inputs = keras.layers.Input(shape=(200,1),name='inputs')
x1 = keras.layers.Conv1D(64,1,padding='same',activation='relu')(inputs)
x1 = tf.keras.layers.MaxPool1D(pool_size=2)(x1)
x1 = keras.layers.Conv1D(64,1,padding='same',activation='relu')(x1)
x1 = tf.keras.layers.MaxPool1D(pool_size=2)(x1)

x2 = keras.layers.Conv1D(64,3,padding='same',activation='relu')(inputs)
x2 = tf.keras.layers.MaxPool1D(pool_size=2)(x2)
x2 = keras.layers.Conv1D(64,3,padding='same',activation='relu')(x2)
x2 = tf.keras.layers.MaxPool1D(pool_size=2)(x2)

x3 = keras.layers.Conv1D(64,9,padding='same',activation='relu')(inputs)
x3 = tf.keras.layers.MaxPool1D(pool_size=2)(x3)
x3 = keras.layers.Conv1D(64,9,padding='same',activation='relu')(x3)
x3 = tf.keras.layers.MaxPool1D(pool_size=2)(x3)

x4 = keras.layers.Conv1D(64,11,padding='same',activation='relu')(inputs)
x4 = tf.keras.layers.MaxPool1D(pool_size=2)(x4)
x4 = keras.layers.Conv1D(64,11,padding='same',activation='relu')(x4)
x4 = tf.keras.layers.MaxPool1D(pool_size=2)(x4)

x = tf.keras.layers.Concatenate()([x1,x2,x3,x4])
x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dense(128,activation='relu')(x)
x = keras.layers.Dense(3)(x)
outputs = keras.layers.Dense(3, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])
model.summary()

# training model
history = model.fit(X_train,y_train,batch_size=16,epochs =100,validation_data=(X_validation,y_validation))

# Preservation of model weights
model_save_path = r'E:\OSR\weight_1-DMFCNN_3fenlei.h5'
model.save(model_save_path)

# Plotting the acc-loss curve
plt.figure(figsize=(12, 4), dpi=200)
ax1 = plt.subplot(1, 2, 1)
plt.plot(history.epoch, history.history.get('accuracy'), label='acc')
plt.plot(history.epoch, history.history.get('validation accuracy'), label='val_acc')
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()
ax2 = plt.subplot(1, 2, 2)
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()

