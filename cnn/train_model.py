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
import pydot

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
print("inputs:", X_train[0].shape)
print("labels:", Y_train.shape)

visible1 = Input(shape=(56, 120, 1))
conv11 = Conv2D(32, kernel_size=4, activation='relu', padding='same')(visible1)
pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
conv12 = Conv2D(16, kernel_size=4, activation='relu', padding='same')(pool11)
pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
drop = Dropout(0.7)(pool12)
flat1 = Flatten()(drop)
#flat1 = Flatten()(pool12)

#merge = concatenate([flat1, flat2, flat3, flat4])
merge = concatenate([flat1])

# interpretation model
hidden1 = Dense(32, activation='relu')(merge)
hidden2 = Dense(32, activation='relu')(hidden1)
output = Dense(2, activation='softmax')(hidden2)
#output = Dense(2)(hidden2)
#class number    
#cnn = Model(inputs=[visible1, visible2, visible3, visible4], outputs=output)
cnn = Model(inputs=[visible1],outputs=output)
print(output)

def train_iteration(lr, epochs):
    print(f"=== Training with lr={lr} for {epochs} epochs ===")
    cnn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    history = cnn.fit(
            X_train, Y_train,
            epochs=epochs, batch_size=128,
            validation_data=(X_test, Y_test),)
    return history 

##cnn.summary()
##keras.utils.plot_model(cnn, "my_model.png", show_shapes=True)
train_iteration(lr=1e-2, epochs=3 )
train_iteration(lr=3e-3, epochs=5 )
train_iteration(lr=1e-3, epochs=5 )
train_iteration(lr=3e-4, epochs=5 )
train_iteration(lr=3e-5, epochs=20 )
train_iteration(lr=1e-5, epochs=10 )
train_iteration(lr=1e-6, epochs=5 )
train_iteration(lr=1e-7, epochs=5 )

# Save this model
os.makedirs(f"{MODEL_DIR}", exist_ok=True)

cnn.save(f"{MODEL_DIR}/{task_name}_accuracy_cnn.h5")
