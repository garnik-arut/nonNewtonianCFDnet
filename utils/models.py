import tensorflow as tf


class NeuralNetwork:

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.channels = input_shape[2]
        self.arch = None
        self.model = None

    def set_architecture(self, sizefilter=(3, 3), stride1=(1, 1), stride2=(1, 1), filter1=None, filter2=None, alpha=None,
                         lamreg=0):
        self.set_input()
        self.conv2d_layer(nfilters=int(filter1 / 2), sizefilter=sizefilter, strides=(1, 1), padding="same",
                          kernel_regularizer=tf.keras.regularizers.l2(lamreg), data_format="channels_last")
        self.leaky_relu(alpha=alpha)
        self.conv2d_layer(nfilters=filter1, sizefilter=sizefilter, strides=stride1, padding="same",
                          kernel_regularizer=tf.keras.regularizers.l2(lamreg))
        self.leaky_relu(alpha=alpha)
        self.conv2d_layer(nfilters=filter2, sizefilter=sizefilter, strides=stride2, padding="same",
                          kernel_regularizer=tf.keras.regularizers.l2(lamreg))
        self.leaky_relu(alpha=alpha)
        self.conv2d_transpose_layer(nfilters=filter2, sizefilter=sizefilter, strides=stride2, padding="same",
                                    kernel_regularizer=tf.keras.regularizers.l2(lamreg))
        self.leaky_relu(alpha=alpha)
        self.conv2d_transpose_layer(nfilters=filter1, sizefilter=sizefilter, strides=stride1, padding="same",
                                    kernel_regularizer=tf.keras.regularizers.l2(lamreg))
        self.leaky_relu(alpha=alpha)
        self.conv2d_transpose_layer(nfilters=self.channels, sizefilter=sizefilter, strides=(1, 1), padding="same",
                                    kernel_regularizer=tf.keras.regularizers.l2(lamreg))
        self.leaky_relu(alpha=alpha)

    def set_architecture_deep(self, sizefilter=(3, 3), stride1=(1, 1), stride2=(1, 1), filter1=None, filter2=None,
                              alpha=None, lamreg=0):
        self.set_input()
        self.conv2d_layer(nfilters=int(filter1 / 2), sizefilter=sizefilter, strides=(1, 1), padding="same",
                          kernel_regularizer=tf.keras.regularizers.l2(lamreg), data_format="channels_last")
        self.leaky_relu(alpha=alpha)
        self.conv2d_layer(nfilters=filter1, sizefilter=sizefilter, strides=stride1, padding="same",
                          kernel_regularizer=tf.keras.regularizers.l2(lamreg))
        self.leaky_relu(alpha=alpha)
        self.conv2d_layer(nfilters=int(filter2 / 2), sizefilter=sizefilter, strides=(1, 1), padding="same",
                          kernel_regularizer=tf.keras.regularizers.l2(lamreg))
        self.leaky_relu(alpha=alpha)
        self.conv2d_layer(nfilters=filter2, sizefilter=sizefilter, strides=stride2, padding="same",
                          kernel_regularizer=tf.keras.regularizers.l2(lamreg))
        self.leaky_relu(alpha=alpha)
        self.conv2d_transpose_layer(nfilters=filter2, sizefilter=sizefilter, strides=stride2, padding="same",
                                    kernel_regularizer=tf.keras.regularizers.l2(lamreg))
        self.leaky_relu(alpha=alpha)
        self.conv2d_transpose_layer(nfilters=int(filter2 / 2), sizefilter=sizefilter, strides=(1, 1), padding="same",
                                    kernel_regularizer=tf.keras.regularizers.l2(lamreg))
        self.leaky_relu(alpha=alpha)
        self.conv2d_transpose_layer(nfilters=filter1, sizefilter=sizefilter, strides=stride1, padding="same",
                                    kernel_regularizer=tf.keras.regularizers.l2(lamreg))
        self.leaky_relu(alpha=alpha)
        self.conv2d_transpose_layer(nfilters=self.channels, sizefilter=sizefilter, strides=(1, 1), padding="same",
                                    kernel_regularizer=tf.keras.regularizers.l2(lamreg))
        self.leaky_relu(alpha=alpha)

    def set_input(self):
        self.inputs = tf.keras.Input(shape=self.input_shape)
        self.arch = self.inputs

    def leaky_relu(self, alpha=None):
        layer = tf.keras.layers.LeakyReLU(alpha=alpha)
        self.arch = layer(self.arch)

    def elu(self, alpha=None):
        layer = tf.keras.layers.Activation(tf.keras.activations.elu)
        self.arch = layer(self.arch)

    def tanh(self):
        layer = tf.keras.layers.Activation(tf.keras.activations.tanh)
        self.arch = layer(self.arch)

    def conv2d_layer(self, nfilters=None, sizefilter=None, strides=None, padding=None, kernel_regularizer=None,
                     data_format=None):
        layer = tf.keras.layers.Conv2D(filters=nfilters,
                                       kernel_size=sizefilter,
                                       strides=strides,
                                       padding=padding,
                                       kernel_regularizer=kernel_regularizer,
                                       data_format=data_format)

        self.arch = layer(self.arch)

    def clear(self):
        tf.keras.backend.clear_session()

    def conv2d_transpose_layer(self, nfilters=None, sizefilter=None, strides=None, padding=None, kernel_regularizer=None,
                               data_format=None):
        layer = tf.keras.layers.Conv2DTranspose(filters=nfilters,
                                                kernel_size=sizefilter,
                                                strides=strides,
                                                padding=padding,
                                                kernel_regularizer=kernel_regularizer,
                                                data_format=data_format)

        self.arch = layer(self.arch)

    def create_model(self):
        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.arch)
