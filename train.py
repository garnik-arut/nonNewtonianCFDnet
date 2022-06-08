import sys
# sys.path.insert(0, 'utils/')

import plotting
import time

import utils.plot as plot
import utils.metrics as metrics
import utils.models as models
import utils.loadDataset as loadDataset
import utils.losses as losses

import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.config.list_physical_devices('GPU'))
    try:
        with tf.device('/device:GPU:0'):
            mirrored_strategy = tf.distribute.MirroredStrategy()

            # training image size (resolution of the coarse-grid data)
            height = 64
            width = 64
            channels = 4

            # load training dataset
            train_dataset_path = './datasets/train/cavity_dataset.h5'
            X_train, Y_train = loadDataset.load_train_dataset(train_dataset_path)

            X_train, Y_train = np.float32(X_train), np.float32(Y_train)

            samples = X_train.shape[0]

            val_dataset_path = './datasets/validation/cavity_dataset.h5'
            X_val, Y_val = loadDataset.load_train_dataset(val_dataset_path)
            X_val, Y_val = np.float32(X_val), np.float32(Y_val)

            input_shape = (height, width, channels)

            # distribute to the number of available gpus
            with mirrored_strategy.scope():
                cnn = models.NeuralNetwork(input_shape)
                cnn.set_architecture_deep(sizefilter=(5, 5),
                                          stride0=(2, 2), stride1=(2, 2), stride2=(4, 4),stride3=(4, 4),
                                          filter0=4, filter1=32, filter2=64, filter3=256, alpha=0.1,
                                          lamreg=0)
                cnn.create_model()
                cnn.model.compile(loss=losses.mse_total, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                                  metrics=['mse', metrics.mse_ux, metrics.mse_nut])

                cnn.model.summary()
            exit()
            batch_sizes = [64]
            epochs = 50
            steps = [1]

            for batch_size in batch_sizes:
                name = "Adam-1e-3-deep-RLROP"
                checkpoint_filepath = './weights/' + name + '-checkpoint-{epoch:02d}'

                callbacks = [ModelCheckpoint(filepath=checkpoint_filepath, monitor='loss', verbose=1,
                                             save_best_only=False, mode='auto', save_freq=10 * samples, save_weights_only=True)]

                callbacks = [ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=10, verbose=1,
                    mode='auto', min_delta=0.01, cooldown=0, min_lr=0
                )]

                # callbacks=[]

                history = cnn.model.fit([X_train], [Y_train],
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        verbose=1,
                                        validation_data=[X_val, Y_val], shuffle=True, callbacks=callbacks)

                cnn.model.save("models/model" + name + ".h5")
                plot.plot_history(history, name, writing=1)

    except RuntimeError as e:
        print(e)


if __name__ == '__main__':
    main()
