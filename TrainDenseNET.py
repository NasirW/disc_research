# -*- coding: utf-8 -*-
"""
Georgios Georgalis
"""
# if used on a non-GUI server ######
import matplotlib

matplotlib.use("Agg")
###################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import h5py

from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dense

# from keras.layers import Flatten
from keras.layers import AveragePooling2D

# from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Conv2DTranspose
# from keras.layers.merge import concatenate
from keras.layers import Dropout

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
import time

# import skimage.transform


# %%
# load the training and val data from already split files
X_tr = np.load("gen/X_tr.npy")
Y_tr = np.load("gen/Y_tr.npy")

X_ts = np.load("gen/X_ts.npy")
Y_ts = np.load("gen/Y_ts.npy")

# X_val = np.load('X_val.npy')
# Y_val = np.load('Y_val.npy')

from keras.utils import to_categorical

Y_tr = to_categorical(Y_tr)
Y_ts = to_categorical(Y_ts)


# %% set-up the DenseNET model
import tensorflow as tf


def H(inputs, num_filters, dropout_rate):
    x = tf.keras.layers.BatchNormalization(epsilon=eps)(inputs)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    x = tf.keras.layers.Conv2D(
        num_filters, kernel_size=(3, 3), use_bias=False, kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    return x


def transition(inputs, num_filters, compression_factor, dropout_rate):
    # compression_factor is the 'θ'
    x = tf.keras.layers.BatchNormalization(epsilon=eps)(inputs)
    x = tf.keras.layers.Activation("relu")(x)
    num_feature_maps = inputs.shape[1]  # The value of 'm'
    x = tf.keras.layers.Conv2D(
        np.floor(compression_factor * num_feature_maps).astype(int),
        kernel_size=(1, 1),
        use_bias=False,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
    )(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    return x


def dense_block(inputs, num_layers, num_filters, growth_rate, dropout_rate):
    for i in range(num_layers):  # num_layers is the value of 'l'
        conv_outputs = H(inputs, num_filters, dropout_rate)
        inputs = tf.keras.layers.Concatenate()([conv_outputs, inputs])
        num_filters += growth_rate  # To increase the number of filters for each layer.
    return inputs, num_filters


input_shape = (80, 80, 3)
num_blocks = 3
num_layers_per_block = 4
growth_rate = 16
dropout_rate = 0.1
compress_factor = 0.5
eps = 1.1e-5

num_filters = 16

inputs = tf.keras.layers.Input(shape=input_shape)
x = tf.keras.layers.Conv2D(
    num_filters,
    kernel_size=(3, 3),
    use_bias=False,
    kernel_initializer="he_normal",
    kernel_regularizer=tf.keras.regularizers.l2(1e-5),
)(inputs)

for i in range(num_blocks):
    x, num_filters = dense_block(
        x, num_layers_per_block, num_filters, growth_rate, dropout_rate
    )
    x = transition(x, num_filters, compress_factor, dropout_rate)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(2)(x)  # 2 Classes
outputs = tf.keras.layers.Activation("softmax")(x)

model = tf.keras.models.Model(inputs, outputs)
model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(lr=0.00001),
    metrics=["acc"],
)
model.summary()

# %% train the model
filepath = "DenseNET_train_loss"

# save the model when val_loss improves during training
checkpoint = ModelCheckpoint(
    "./trained_models/" + filepath + ".hdf5",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="auto",
)

# save training progress in a .csv
csvlog = CSVLogger("./trained_models/" + filepath + "_train_log.csv", append=True)
# stop training if no improvement has been seen on val_loss for a while
early_stopping = EarlyStopping(monitor="val_loss", min_delta=0, patience=300)
batch_size = 32
epochs = 300

history = model.fit(
    X_tr,
    Y_tr,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_ts, Y_ts),
    verbose=2,
    initial_epoch=0,
    callbacks=[checkpoint, csvlog, early_stopping],
)



# Plotting the loss curves
plt.figure(figsize=(12, 6))

# Plotting the training loss
plt.plot(history.history['loss'], label='Training Loss')

# Plotting the validation loss
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.savefig('results/Training and validation loss.png')
plt.show()