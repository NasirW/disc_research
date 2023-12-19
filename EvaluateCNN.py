# -*- coding: utf-8 -*-
"""
Georgios Georgalis
"""
# if used on a non-GUI server ######
import matplotlib

# matplotlib.use('tkagg')
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

X_ts = np.load("gen/X_ts.npy")
Y_ts = np.load("gen/Y_ts.npy")

# X_val = np.load('X_val.npy')
# Y_val = np.load('Y_val.npy')

X_tr = np.load("gen/X_tr.npy")
Y_tr = np.load("gen/Y_tr.npy")

Ntest = len(X_ts)


# %% set-up the model

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
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
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
dropout_rate = 0.2
compress_factor = 0.5
eps = 1.1e-5

num_filters = 16

inputs = tf.keras.layers.Input(shape=input_shape)
x = tf.keras.layers.Conv2D(
    num_filters,
    kernel_size=(3, 3),
    use_bias=False,
    kernel_initializer="he_normal",
    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
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
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    metrics=["acc"],
)
model.summary()

# %% get predicted masks for test set
# model.load_weights('DenseNET_train_val3.hdf5')


# training_hist = pd.read_csv('DenseNET_train_val3_train_log.csv')
# training_hist['dloss'] = training_hist['loss'].diff()
# training_hist['dval_loss'] = training_hist['val_loss'].diff()
# training_hist['dloss'][0] = 0
# training_hist['dval_loss'][0] = 0


# plt.plot(training_hist['loss'], label='Train Loss', c = 'blue')
# #plt.plot(training_hist['dloss'], 'b--', label='Train Loss Change')
# plt.plot(training_hist['val_loss'], label='Test/Validation Loss', c = 'red')
# #plt.plot(training_hist['dval_loss'], 'r--', label='Test/Validation Change')
# plt.legend()
# plt.xlabel(r'Epoch', fontsize = 20)
# plt.ylabel(r'Binary Crossentropy', fontsize = 20)
# plt.grid(True, which = 'both')
# plt.show()


# %%Testset

# Y_ts_hat = model.predict(X_ts,batch_size=1)
# Y_ts_hat_df = pd.DataFrame(data = Y_ts_hat)
# Y_ts_hat_df['Predicted label'] =  Y_ts_hat_df.idxmax(axis=1)
# Y_ts_hat_df['True label'] = Y_ts

# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# y_true = Y_ts_hat_df['True label']
# y_pred = Y_ts_hat_df['Predicted label']
# cm = confusion_matrix(y_true, y_pred)

# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.show()

# accuracy_ts_val = (cm[0,0]+cm[1,1])/130
# accuracy_ci_ts_val = 1.96*np.sqrt(accuracy_ts_val*(1-accuracy_ts_val)/130)

# Specificity = cm[0,0]/(cm[0,0]+cm[0,1])
# Sensitivity = cm[1,1]/(cm[1,0]+cm[1,1])

img_folder_sensitive = "./data/chemo_res/Sensitive"  # Folder that contains the sensitive biopsied RGB images
img_folder_resistant = "./data/chemo_res/Resistant"  # Folder that contains the resistant biopsied RGB images

image_height = 80
image_width = 80

import os

# load filenames
img_filenames_sensitive = np.array(sorted(os.listdir(img_folder_sensitive)))  # sort
img_filenames_resistant = np.array(sorted(os.listdir(img_folder_resistant)))  # sort

d1 = np.transpose(
    np.vstack([img_filenames_sensitive, np.zeros(np.size(img_filenames_sensitive))])
)
d2 = np.transpose(
    np.vstack([img_filenames_resistant, np.ones(np.size(img_filenames_resistant))])
)
d = np.vstack([d1, d2])

labels = pd.DataFrame(data=d, columns=["image_name", "label"])


wells = labels["image_name"]
Nimages = np.size(wells)  # Number of images

# Empty arrays for RGB images (X)
X = np.zeros(shape=(Nimages, image_height, image_width, 3), dtype="float32")
Y = np.zeros(shape=Nimages)

import cv2

i = 0
for w in range(np.size(labels, 0)):
    print("loading image ", w)
    img_file = labels.iloc[w, 0]
    if labels.iloc[w, 1] == "0.0":
        img = cv2.imread(img_folder_sensitive + "/" + img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Y[i] = 0
    else:
        img = cv2.imread(img_folder_resistant + "/" + img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Y[i] = 1
    # normalize pixels
    img = img / img.max()
    X[i, :, :, :] = img

    i = i + 1

Y_hat = model.predict(X_ts, batch_size=1)
Y_hat_df = pd.DataFrame(data=Y_hat)
Y_hat_df["Predicted label"] = Y_hat_df.idxmax(axis=1)
Y_hat_df["True label"] = Y_ts

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

y_true2 = Y_hat_df["True label"]
y_pred2 = Y_hat_df["Predicted label"]
cm2 = confusion_matrix(y_true2, y_pred2)

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true2, y_pred2)
auc_keras = auc(fpr_keras, tpr_keras)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

disp = ConfusionMatrixDisplay(confusion_matrix=cm2)  # Remove the 'ax=ax1' argument
disp.plot(ax=ax1)
ax1.set_title('Confusion Matrix')

ax2.plot([0, 1], [0, 1], "k--")
ax2.plot(fpr_keras, tpr_keras, label="Keras (area = {:.3f})".format(auc_keras))
ax2.set_xlabel("False positive rate")
ax2.set_ylabel("True positive rate")
ax2.set_title("ROC Curve")
ax2.legend(loc="best")

plt.savefig("confusion_matrix_roc.png")
# plt.show()
