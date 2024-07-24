# Allow importing from one level higher
import os
import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import pandas as pd
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools

import h5py
from h5py import File as HDF5File

import enum

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

from keras.layers import Lambda, Input
from keras.layers import Dropout, Flatten, Dense
import keras.backend as K
from keras.models import Sequential, Model 
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import concatenate

from config import DATA_DIR, MODEL_DIR
task_name = sys.argv[1]

with open(f"{DATA_DIR}/processed/cnn/{task_name}_X_train.pkl", "rb") as fin:
    X_train = pickle.load(fin)
with open(f"{DATA_DIR}/processed/cnn/{task_name}_Y_train.pkl", "rb") as fin:
    Y_train = pickle.load(fin)
with open(f"{DATA_DIR}/processed/cnn/{task_name}_X_test.pkl", "rb") as fin:
    X_test = pickle.load(fin)
with open(f"{DATA_DIR}/processed/cnn/{task_name}_Y_test.pkl", "rb") as fin:
    Y_test = pickle.load(fin)

print("Training shapes:")
print(" inputs:", X_train[0].shape)
print("labels:", Y_train.shape)

visible1 = Input(shape=(56, 120, 1))
conv11 = Conv2D(32, kernel_size=4, activation='relu', padding='same')(visible1)
pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
conv12 = Conv2D(16, kernel_size=4, activation='relu', padding='same')(pool11)
pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
flat1 = Flatten()(pool12)
print ('cov11.shape', conv11.shape)
print ('pool1.shape', pool11.shape)
print ('cov12.shape', conv12.shape)
print ('pool2.shape', pool12.shape)

###visible2 = Input(shape=(4, 128, 1))
###conv21 = Conv2D(32, kernel_size=4, activation='relu', padding='same')(visible2)
###pool21 = MaxPooling2D(pool_size=(2, 2))(conv21)
###conv22 = Conv2D(16, kernel_size=4, activation='relu', padding='same')(pool21)
###pool22 = MaxPooling2D(pool_size=(2, 2))(conv22)
###flat2 = Flatten()(pool22)
###print ('cov21.shape', conv21.shape)
###print ('poo21.shape', pool21.shape)
###print ('cov22.shape', conv22.shape)
###print ('poo22.shape', pool22.shape)
###
###visible3 = Input(shape=(16, 16, 1))
###conv31 = Conv2D(32, kernel_size=4, activation='relu', padding='same')(visible3)
###pool31 = MaxPooling2D(pool_size=(2, 2))(conv31)
###conv32 = Conv2D(16, kernel_size=4, activation='relu', padding='same')(pool31)
###pool32 = MaxPooling2D(pool_size=(2, 2))(conv32)
###flat3 = Flatten()(pool32)
###print ('cov31.shape', conv31.shape)
###print ('poo31.shape', pool31.shape)
###print ('cov32.shape', conv32.shape)
###print ('poo32.shape', pool32.shape)
###
###visible4 = Input(shape=(16, 8, 1))
###conv41 = Conv2D(32, kernel_size=4, activation='relu', padding='same')(visible4)
###pool41 = MaxPooling2D(pool_size=(2, 2))(conv41)
###conv42 = Conv2D(16, kernel_size=4, activation='relu', padding='same')(pool41)
###pool42 = MaxPooling2D(pool_size=(2, 2))(conv42)
###flat4 = Flatten()(pool32)
###print ('cov41.shape', conv41.shape)
###print ('poo41.shape', pool41.shape)
###print ('cov42.shape', conv42.shape)
###print ('poo42.shape', pool42.shape)
###

#merge = concatenate([flat1, flat2, flat3, flat4])
merge = concatenate([flat1])

# interpretation model
hidden1 = Dense(32, activation='relu')(merge)
hidden2 = Dense(32, activation='relu')(hidden1)
output = Dense(2)(hidden2)
#class number    
#cnn = Model(inputs=[visible1, visible2, visible3, visible4], outputs=output)
cnn = Model(inputs=[visible1],outputs=output)

def train_iteration(lr, epochs):
    print(f"=== Training with lr={lr} for {epochs} epochs ===")
    cnn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    return cnn.fit(
        X_train, Y_train,
        epochs=epochs, batch_size=128,
        validation_data=(X_test, Y_test),
    )

train_iteration(lr=1e-2, epochs=3 )
train_iteration(lr=2e-3, epochs=5 )
train_iteration(lr=1e-3, epochs=5 )
train_iteration(lr=2e-4, epochs=5 )
train_iteration(lr=2e-5, epochs=30 )
train_iteration(lr=1e-5, epochs=10 )

#### draw confusion matrix
###Y_pred =np.argmax(cnn.predict(X_test,batch_size = 10),axis=1)
###cm = confusion_matrix(Y_test, Y_pred, normalize = 'true' )
###disp = ConfusionMatrixDisplay(confusion_matrix = cm, 
###   #display_labels=["K_nu","pi_2pi0_nu","pi_nu","pi_pi0_gam_nu","pi_pi0_nu","signal"]) 
###    display_labels=["pi_pi0_nu","signal"])
###disp.plot(cmap = "Blues", values_format = '.4f')
###plt.yticks(rotation=90)
###plt.tight_layout()
###plt.savefig("ConfusionMatrix.pdf")

# Save this model
os.makedirs(f"{MODEL_DIR}", exist_ok=True)

cnn.save(f"{MODEL_DIR}/{task_name}_cnn.h5")
