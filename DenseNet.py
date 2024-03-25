import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

class DenseNet:
    def __init__(self, X_tr, Y_tr, X_ts, Y_ts):
        self.X_tr = X_tr
        self.Y_tr = to_categorical(Y_tr)
        self.X_ts = X_ts
        self.Y_ts = to_categorical(Y_ts)
        self.model = None
        
        self.build_model()

    def build_model(self):
        input_shape = (80, 80, 3)
        num_blocks = 3
        num_layers_per_block = 4
        growth_rate = 16
        dropout_rate = 0.1
        compress_factor = 0.5
        self.eps = 1.1e-5
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
            x, num_filters = self.dense_block(
                x, num_layers_per_block, num_filters, growth_rate, dropout_rate
            )
            x = self.transition(x, num_filters, compress_factor, dropout_rate)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(2)(x)  # 2 Classes
        outputs = tf.keras.layers.Activation("softmax")(x)

        self.model = tf.keras.models.Model(inputs, outputs)
        self.model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(lr=0.00001),
            metrics=["acc"],
        )

    def dense_block(self, inputs, num_layers, num_filters, growth_rate, dropout_rate):
        for i in range(num_layers):
            conv_outputs = self.H(inputs, num_filters, dropout_rate)
            inputs = tf.keras.layers.Concatenate()([conv_outputs, inputs])
            num_filters += growth_rate
        return inputs, num_filters

    def transition(self, inputs, num_filters, compression_factor, dropout_rate):
        x = tf.keras.layers.BatchNormalization(epsilon=self.eps)(inputs)
        x = tf.keras.layers.Activation("relu")(x)
        num_feature_maps = inputs.shape[1]
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

    def H(self, inputs, num_filters, dropout_rate):
        x = tf.keras.layers.BatchNormalization(epsilon=self.eps)(inputs)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
        x = tf.keras.layers.Conv2D(
            num_filters, kernel_size=(3, 3), use_bias=False, kernel_initializer="he_normal"
        )(x)
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
        return x

    def train(self, filepath, epochs=300, batch_size=128):
        checkpoint = ModelCheckpoint(
            filepath + ".hdf5",
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="auto",
        )
        csvlog = CSVLogger(filepath + "_train_log.csv", append=True)
        early_stopping = EarlyStopping(monitor="val_loss", min_delta=0, patience=300)

        history = self.model.fit(
            self.X_tr,
            self.Y_tr,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_ts, self.Y_ts),
            verbose=2,
            initial_epoch=0,
            callbacks=[checkpoint, csvlog, early_stopping],
        )

        return history

    def plot_loss(self, history):
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')

        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('results/Training and validation loss.png')
        plt.show()

    def plot_loss_from_csv(self, filepath):
        csv_data = pd.read_csv(filepath + "_train_log.csv")
        plt.figure(figsize=(12, 6))
        plt.plot(csv_data['epoch'], csv_data['loss'], label='Training Loss from CSV')
        plt.plot(csv_data['epoch'], csv_data['val_loss'], label='Validation Loss from CSV')
        plt.title('Training and Validation Loss (CSV)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('results/Training and validation loss(from csvlog).png')
        plt.show()