# importing

from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend
import tensorflow as tf
class ResNet:
    @staticmethod
    def residual_module(data, num_filters, stride, chandim, reduce_spacial_dim=False, regularization=.0001,
                        batchnorm_Epsilon=2e-5, batchnorm_Momentum=.9):
        shortcut =data
        # batchnorm->Relu->convolution
        # first block has a 1x1 layer with k/4
        batchnorm_1 = BatchNormalization(axis=chandim,
                                         epsilon=batchnorm_Epsilon, momentum=batchnorm_Momentum)(data)
        activation_1 = Activation("relu")(batchnorm_1)
        convolution_1 = Conv2D(
            int(num_filters/4), (1, 1) ,use_bias=False, padding="same", kernel_regularizer=l2(regularization))(activation_1)
        # second block this one has a 3x3 layer with k/4 filters
        batchnorm_2 = BatchNormalization(axis=chandim,
                                         epsilon=batchnorm_Epsilon, momentum=batchnorm_Momentum)(convolution_1)
        activation_2 = Activation("relu")(batchnorm_2)
        convolution_2 = Conv2D(
            int(num_filters/4), (3, 3), use_bias=False, padding="same", kernel_regularizer=l2(regularization))(activation_2)
        # third block this one has a 1x1 layer with k filters
        batchnorm_3 = BatchNormalization(axis=chandim,
                                         epsilon=batchnorm_Epsilon, momentum=batchnorm_Momentum)(convolution_2)
        activation_3 = Activation("relu")(batchnorm_3)
        convolution_3 = Conv2D(
            num_filters, (1, 1), use_bias=False, kernel_regularizer=l2(regularization))(activation_3)
        if reduce_spacial_dim:
            shortcut = Conv2D(num_filters, (1, 1), strides=stride,
                              use_bias=False, padding="same", kernel_regularizer=l2(regularization))(activation_1)
        holder = add([convolution_3, shortcut])
        return holder

    @staticmethod
    def build(width, height, depth, classes, stages, filters, regularization=.0001,
              batchnorm_Epsilon=2e-5, batchnorm_Momentum=0.9):
        print("--building--")
        inputshape = (height, width, depth)
        chandim = -1
        inputs = Input(shape=inputshape)
        # apply batch normalization
        holder = BatchNormalization(
            axis=chandim, epsilon=batchnorm_Epsilon, momentum=batchnorm_Momentum)(inputs)
        # apply convolution
        # conv->batchnorm->activation->pooling
        holder = Conv2D(filters[0], (5, 5),input_shape=(200,200,3),use_bias=False,
                        padding="same", kernel_regularizer=l2(regularization))(holder)
        holder = BatchNormalization(
            axis=chandim, epsilon=batchnorm_Epsilon, momentum=batchnorm_Momentum)(holder)
        holder = Activation("relu")(holder)
        holder = ZeroPadding2D((1, 1))(holder)
        holder = MaxPooling2D((3, 3), strides=(2, 2))(holder)
        for i in range(0, len(stages)):
            stride = (1, 1)
            holder = ResNet.residual_module(
                holder, filters[i], stride, chandim, reduce_spacial_dim=True)
            for j in range(0, stages[i]-1):
                holder = ResNet.residual_module(
                    holder, filters[i+1], (1, 1), chandim, reduce_spacial_dim=True)
        holder = BatchNormalization(
            axis=chandim, epsilon=batchnorm_Epsilon, momentum=batchnorm_Momentum)(holder)
        holder = Activation("relu")(holder)
        holder = AveragePooling2D((8, 8))(holder)
        holder = Flatten()(holder)
        holder = Dense(classes, kernel_regularizer=l2(regularization))(holder)
        holder = Activation("softmax")(holder)
        model = Model(inputs, holder, name="resnet")
        return model
