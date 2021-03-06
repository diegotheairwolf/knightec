# -*- coding: utf-8 -*-
"""Knightec_CNN_LeNet_v2z.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_nEBlnCJR1JkWRwnVUFlKB4iodi_R5kG

Using Enric's data with CNN LeNet model
- sampling rate is 5kHz
- 


- will use 12kHz files since the normal data is in 12kHz
- images are 16*16 since 12kHz have less samples
- loading from CWRU_16.pickle
- normalizing data between [0,1] instead of [0,255]
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy import signal
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

import os
from datetime import datetime

from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split

import copy

from sklearn.utils import class_weight

import knightec_py3

## GLOBAL SEED ##                                                   
tf.random.set_seed(1)
np.random.seed(1)
print(tf.version.VERSION)

M = 16
sample_rate = 5000



# work directory of all data
work_dir = os.getcwd()
rdir = os.path.join(os.path.expanduser(work_dir), 'Datasets')

# load all data previously pickled (preprocess.py)
filename = 'knightec_100_5000_{}.pickle'.format(M)
with open(os.path.join(rdir, filename), 'rb') as f:
  data = pickle.load(f)

thesis_dir = os.getcwd()
current_dir = "4_Knightec_CNN_LeNet"
file_name = "Knightec_CNN_LeNet_1"
rdir = os.path.join(os.path.expanduser(thesis_dir), current_dir)

"""Obtain normal data and split for each of the different loads."""

rpm1797 = 0
rpm1772 = 1
rpm1750 = 2
rpm1730 = 3

ball7 = 0
ir7 = 1

"""For this notebook, I will use 12kHz data at 1797 RPM."""

data_copy = copy.deepcopy(data)

data.X_v2z_train

"""Split training set into train / validation.
NOTE! Will skip this step as it is automatically done when training the model.
"""

# Split training data into training and validatiom
print(data.X_v2z_train.shape)
data.X_v2z_train, data.X_v2z_val, data.y_train, data.y_val = train_test_split(data.X_v2z_train, data.y_train, 
                                                                      test_size=0.2, random_state=42, 
                                                                      shuffle=False, stratify=None)
print(data.X_v2z_train.shape)
print(data.X_v2z_val.shape)

slice_size = M*M
T = 1.0 / sample_rate
N = (2 - 0) * sample_rate
time = np.linspace(0, 2, N)

# Use DataFrame for better manipulation
#data_columns = ["0.007B", "0.007IR", "0.007OR12", "0.007OR3", "0.007OR6", "0.014B", "0.014IR", "0.014OR6", "0.021B", "0.021IR", "0.021OR12", "0.021OR3", "0.021OR6", "Normal"]
df = pd.DataFrame(data.X_v2z_train[0])
df.index.name = "Cycles"
print(df)

def to_image(df, M):
    """
    single image conversion from dataframe[0:M*M] to M*M ndarray
    :param df:
    :param M:
    :return:
    """
    P = df.values.reshape(M, M)
    #P = np.round((P-np.min(df.values)/(np.max(df.values)-np.min(df.values)))*255)
    P = P-np.min(df.values) / (np.max(df.values)-np.min(df.values))
    return P

# Convert dataframe element to image for testing purposes
data_to_convert = df[0:(M*M)]
data_image = to_image(data_to_convert, M)
# plt.imshow(data_image, cmap='gray')
plt.imshow(data_image)
plt.show()

data_image

def to_images(data):
    """
    Converts time-series data into normalized images
    :param data:
    :return:
    """
    min_val = np.amin(data, axis=1).reshape(-1, 1)
    max_val = np.amax(data, axis=1).reshape(-1, 1)
    #P = np.around(((data-min_val) / (max_val-min_val)) * 255)
    #P = (data-min_val) / (max_val-min_val)
    P = data-min_val / (max_val-min_val)
    return P

def show_images(images, labels):
    """
    Creates a collage all class images
    :param images: all images data. ndarray of 64x64 elements
    :param labels: labels of images. 0-13
    :return: none
    """

    image_index, unique_img = [], []
    for i in range(0, len(data.labels)):
        image_index.append(labels.index(i))
        unique_img.append(images[labels.index(i)].squeeze(axis=2))

    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, int(len(data.labels)/2)),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    for i, (ax, im) in enumerate(zip(grid, unique_img)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(data.labels[i])

    plt.show()

# convert all time-series slices to images

train_X = np.array([im.reshape(M, M) for im in to_images(data.X_v2z_train)])
test_X = np.array([im.reshape(M, M) for im in to_images(data.X_v2z_test)])
val_X = np.array([im.reshape(M, M) for im in to_images(data.X_v2z_val)])

# add channel for grayscale
train_X = train_X[..., np.newaxis]
test_X = test_X[..., np.newaxis]
val_X = val_X[..., np.newaxis]

# #display some of the images
# show_images(train_X, data.y_train)

train_X[0]

def OneHot(y):
    """
        input:  y (n) = labels
        output: onehot (Kxn) = one-hot representation of labels
    """
    onehot = np.zeros((len(data.labels), len(y)))
    for idx, val in enumerate(y):
        onehot[val][idx] = 1
    # print("oneshot {}".format(onehot))
    return onehot

# convert labels to one-hot encoding
train_y = OneHot(data.y_train).swapaxes(1, 0)
test_y = OneHot(data.y_test).swapaxes(1, 0)
val_y = OneHot(data.y_val).swapaxes(1, 0)

# ------------------------------------------------
# C. Zero-Padding Calculation

def cal_zero_padding(M, N, F, S):
    """
    The number of padding on left PL and on right PR can be calculated by the following equations.
    :param M: input size
    :param N: output size
    :param F: filter width
    :param S: stride
    :return: PL - left padding
            PR - right padding

    """
    N = np.ceil(M/S)
    PT = (N-1) * S + F - M
    PL = np.floor(PT/2)
    PR = PT - PL

    return PT, PL, PR




# ------------------------------------------------
# B. CNN Model

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(np.argmax(train_y, axis=1)),
                                                 np.argmax(train_y, axis=1))
class_weight_dict = dict(enumerate(class_weights))

# ------------------------------------------------
# B. CNN Model
FD1 = 2560
FD2 = 768

def get_compiled_model():
    """
    model based on LeNet-5
    activation function is tanh
    :return:
    """
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=5, input_shape=(M, M, 1), activation="tanh", padding="same", name="L1"))
    model.add(layers.MaxPool2D(pool_size=2, name="L2"))
    model.add(layers.Conv2D(filters=64, kernel_size=3, activation="tanh", padding="same", name="L3"))
    model.add(layers.MaxPool2D(pool_size=2, name="L4"))
    model.add(layers.Conv2D(filters=128, kernel_size=3, activation="tanh", padding="same", name="L5"))
    model.add(layers.MaxPool2D(pool_size=2, name="L6"))
    model.add(layers.Conv2D(filters=256, kernel_size=3, activation="tanh", padding="same", name="L7"))
    model.add(layers.MaxPool2D(pool_size=2, name="L8"))
    model.add(layers.Flatten())
    model.add(layers.Dense(2560, activation='tanh'))
    model.add(layers.Dense(768, activation='tanh'))
    model.add(layers.Dense(len(data.labels), activation='softmax'))
    print(model.summary())

    return model

model = get_compiled_model()

# Set hyperparameters
batch_size = 32
learning_rate = 2e-4 
epochs = 200

# Set optimizer
optimizer = keras.optimizers.Adam(learning_rate)


# Callbacks, early stopping and best model restoration
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', 
                          verbose=1, patience=30, mode='min', 
                          restore_best_weights=True)

# create checkpoints to save model after each iteration to be used later
# https://www.tensorflow.org/tutorials/keras/save_and_load

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = os.path.join(os.path.join(rdir, file_name), "training_1/cp-{epoch:04d}.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_best_only=True,
                                                 verbose=1,
                                                 monitor="val_loss",
                                                 save_freq=5*batch_size)

model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mean_squared_error'])

# Save the weights using the 'checkpoint_path' format
model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, 
          callbacks=[early_stopping, cp_callback], 
          steps_per_epoch = int(len(train_y) / batch_size), 
          validation_data=(val_X, val_y),
          class_weight=class_weights)
# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.

# # Save the entire model to a HDF5 file.
# # The '.h5' extension indicates that the model should be saved to HDF5.
# # current date and time
# now = datetime.now()
# timestamp = datetime.timestamp(now)
# model_name = timestamp
# model.save('{}.h5'.format(timestamp))

# list all data in history
print(history.history.keys())

# Plot results
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.ylim([0, 0.1])
plt.legend(loc='lower right')

# Predict validation data
val_y_pred = np.argmax(model.predict(val_X), axis=1)
print("Validation set prediction:")

conf_matrix_val = confusion_matrix(val_y.argmax(axis=1), val_y_pred, labels=[i for i, j in enumerate(data.labels)])
print(conf_matrix_val)
class_report_val = classification_report(val_y.argmax(axis=1), val_y_pred, digits=3)
print(class_report_val)

# Predict test data
test_y_pred = np.argmax(model.predict(test_X), axis=1)
print("Test set prediction:")
conf_matrix_test = confusion_matrix(test_y.argmax(axis=1), test_y_pred, labels=[i for i, j in enumerate(data.labels)])
print(conf_matrix_test)
class_report_test = classification_report(test_y.argmax(axis=1), test_y_pred, digits=3)
print(class_report_test)

# Predict test data
test_y_pred = np.argmax(model.predict(test_X), axis=1)
print("Test set prediction:")
conf_matrix_test = confusion_matrix(test_y.argmax(axis=1), test_y_pred, 
                                    labels=[i for i, j in enumerate(data.labels)])
print(conf_matrix_test)
class_report_test = classification_report(test_y.argmax(axis=1), test_y_pred, digits=3)
print(class_report_test)
multi_conf_matrix_test = multilabel_confusion_matrix(test_y.argmax(axis=1), test_y_pred, 
                                                     labels=[i for i, j in enumerate(data.labels)])

# https://www.kaggle.com/agungor2/various-confusion-matrix-plots

df_cm = pd.DataFrame(conf_matrix_val, index=range(len(data.labels)),
                     columns=range(len(data.labels)))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True, fmt='d', xticklabels=data.labels, yticklabels=data.labels)

df_cm = pd.DataFrame(conf_matrix_test, index=range(len(data.labels)),
                     columns=range(len(data.labels)))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True, fmt='d', xticklabels=data.labels, yticklabels=data.labels)

# The evaluate() method - gets the loss statistics
model.evaluate(test_X, test_y, )