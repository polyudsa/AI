# coding: utf-8
"""
mlp class

"""
import os
import random

import keras
import numpy as np
from keras.layers import Activation, Dropout, BatchNormalization, Dense, Input, Flatten
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from keras.optimizer_v2.rmsprop import RMSprop
from keras.datasets import mnist

from keras.utils.vis_utils import pydot
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, CSVLogger
import matplotlib.pyplot as plt

PATH = str(os.path.abspath(__file__))

class MLP():
    def __init__(self, dense1=512, dense2=512,
                 drop1=0.3, drop2=0.2,
                 batch_size=128,
                 activation='relu',
                 opt='Adam'):

        self.dense1 = dense1
        self.dense2 = dense2
        self.drop1 = drop1
        self.drop2 = drop2
        self.activation = activation
        self.batch_size = batch_size
        self.validation_split = 0.2
        self.num_classes = 10
        self.opt_name = opt
        self.activation_name = activation

        if opt == 'SGD1':
            print("optimizer is SGD lr = 0.01")
            self.opt = SGD(lr=0.01)
        elif opt == 'SGD2':
            print("optimizer is SGD lr = 0.001")
            self.opt = SGD(lr=0.001)
        elif opt == 'Adam':
            print("optimizer is Adam")
            self.opt = Adam()
        else:
            print("optimizer is RMSprop")
            self.opt = RMSprop()

        # load mnist data
        print(" load mnist data")
        self.x_train, self.x_test, self.y_train, self.y_test = self.load_mnist_data()
        # build mlp model
        print("  build mlp model")
        self.model = self.mlp_model()

        params = """
        optimizer:\t{0}
        dense1:\t{1}
        dense2:\t{2}
        drop1:\t{3}
        drop2:\t{4}
        activation:\t{5}
        batch_size:\t{6}
        """.format(self.opt,
                   self.dense1,
                   self.dense2,
                   self.drop1,
                   self.drop2,
                   self.activation,
                   self.batch_size
                   )
        print(params)

    def plot_history(self, history):
        plt.clf()
        plot_name = "optimizer:{0}_activation:{1}_batchSize{2}_".format(self.opt_name,self.activation_name,self.batch_size)
        fig, axs = plt.subplots(2, 1)
        plt.subplots_adjust(wspace=0, hspace =0.5)
        axs[0].plot(history.history['accuracy'])
        axs[0].plot(history.history['val_accuracy'])
        axs[0].set_title(self.activation_name+" "+self.opt_name+" model accuracy")
        axs[0].set_xlabel("epoch")
        axs[0].set_ylabel("accuracy")
        axs[0].legend(['accuracy', 'val_accuracy'], loc='lower right')

        epochs = np.arange(0, len(history.history["loss"]))
        axs[1].plot(epochs - 2.5,history.history['loss'])
        axs[1].plot(epochs,history.history['val_loss'])
        axs[1].set_title(self.activation_name+" "+self.opt_name+" model loss")
        axs[1].set_xlabel("epoch")
        axs[1].set_ylabel("loss")
        axs[1].legend(['loss', 'val_loss'], loc='lower right')
        picture_name = plot_name+str(random.randint(1,100))+".png"
        file_name = PATH.replace("mlp.py", "images/mnist/"+picture_name)
        plt.savefig(file_name)
        plt.show()

    def load_mnist_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(60000, 784).astype('float32') / 255
        x_test = x_test.reshape(10000, 784).astype('float32') / 255
        y_train = keras.utils.np_utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.np_utils.to_categorical(y_test, self.num_classes)

        return x_train, x_test, y_train, y_test

    def mlp_model(self):
        model = Sequential()
        model.add(Dense(self.dense1, input_shape=(784,)))
        model.add(Activation(self.activation))
        # model.add(Dropout(self.drop1))

        model.add(Dense(self.dense2))
        model.add(Activation(self.activation))
        # model.add(Dropout(self.drop2))

        model.add(Dense(10))
        model.add(Activation('softmax'))

        return model

    def train(self):
        # early_stopping = EarlyStopping(patience=0, verbose=1)

        csv_logger = CSVLogger(PATH.replace("mlp.py", "data/") + 'training_log.csv', separator=',', append=True)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.opt,
                           metrics=['accuracy'])
        self.model.summary()
        filename_path = PATH.replace("mlp.py", "images/")
        plot_model(self.model, to_file=filename_path + 'mlp.png', show_layer_names=True, show_shapes=True)

        history = self.model.fit(self.x_train, self.y_train,
                                 batch_size=self.batch_size,
                                 epochs=20,
                                 validation_split=self.validation_split,
                                 verbose=0,
                                 validation_data=(self.x_test, self.y_test))
                                 # callbacks=[early_stopping, csv_logger])
        self.plot_history(history)

    def mlp_evaluate(self):
        self.train()

        evaluate = self.model.evaluate(self.x_test, self.y_test,
                                       batch_size=self.batch_size, verbose=0)
        return evaluate
