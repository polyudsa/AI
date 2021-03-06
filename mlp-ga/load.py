# coding: utf-8

"""
加载数据的类
数据整形类
"""

import numpy as np
import csv
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class ImageDataGenerator(object):
    def __init__(self, args):
        self.path = args.datasetpath
        self.seq_length = args.seqlength
        self.strides = args.strides
        self.batch_size = args.batchsize
        self.imgsize = args.imgsize
        self.reset()

    def reset(self):
        """ reset data load list """
        print("reset data load list")
        self.X = []
        self.Y = []
        self.X_data = []
        self.Y_data = []

    def flow_from_directory(self):
        while True:
            with open(self.path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)

                for row in reader:
                    self.Y_data.append(int(row[1]))
                    img = load_img(row[0], target_size=(self.imgsize, self.imgsize))
                    img_array = img_to_array(img)
                    x = (img_array / 255.).astype(np.float32)
                    print("x.shape", x.shape)
                    self.X_data.append(x)

                    # 存储batch size个数后，格式化数据后，返回
                    if len(self.Y_data) == self.batch_size * self.seq_length:

                        """ data format """
                        length_of_sequence = len(self.Y_data)
                        for i in range(0, length_of_sequence - self.seq_length + 1, self.strides):
                            self.X.append(self.X_data[i: i + self.seq_length])

                            # Y_data 数据格式
                            if self.Y_data[i] == 1:
                                self.Y_data[i] = 0

                            print("Y_data list: ", self.Y_data[i:i + self.seq_length])
                            if 1 in self.Y_data[i: i + self.seq_length]:
                                print("this data include shot")
                                self.Y.append(1)
                            else:
                                print("no include shot")
                                self.Y.append(0)

                        X_train = np.array(self.X).reshape(len(self.X), self.seq_length, self.imgsize, self.imgsize, 3)
                        Y_train = np.array(self.Y).reshape(len(self.Y), 1)
                        self.reset()
                        yield X_train, Y_train


def load_csv_data():
    X_data = []
    Y_data = []
    X = []
    Y = []
    seq_length = 10
    strides = 7
    datasetpath = ''

    # load csv file
    with open(datasetpath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            Y_data.append(int(row[1]))

            img = load_img(row[0], target_size=(64, 64))
            img_array = img_to_array(img)
            x = (img_array / 255.).astype(np.float32)
            # print("x.shape", x.shape)
            X_data.append(x)

    """ data format """
    length_of_sequence = len(Y_data)
    for i in range(0, length_of_sequence - seq_length + 1, strides):
        X.append(X_data[i: i + seq_length])

        # Y_data 数据格式
        if Y_data[i] == 1:
            Y_data[i] = 0

        # print("Y_data list: ", Y_data[i:i+seq_length])
        if 1 in Y_data[i: i + seq_length]:
            # print("this data include shot")
            Y.append(1)
        else:
            # print("no include shot")
            Y.append(0)

    # convert np.array
    X = np.array(X).reshape(len(X), seq_length, 64, 64, 3)
    Y = np.array(Y).reshape(len(Y), 1)
    print("convert!!!!!!!!!!!!")
    print(X.shape)
    print(Y.shape)

    """ split train data/ validation data """
    train_len = int(len(Y) * 0.8)
    validation_len = len(Y) - train_len
    X_train, X_valid, Y_train, Y_valid = \
        train_test_split(X, Y, test_size=validation_len)

    return X_train, X_valid, Y_train, Y_valid


class Load_Feature_Data():

    def __init__(self, args):

        self.X_data = []
        self.Y_data = []
        self.X_ = []
        self.Y_ = []
        self.feature_length = args.featurelength
        self.seq_length = args.seqlength
        self.stride = args.stride
        self.datasetpath = args.datasetpath

    def load(self):

        with open(self.datasetpath, 'r') as f:

            reader = csv.reader(f)
            header = next(reader)

            for row in reader:
                self.Y_data.append(int(row[1]))

                feaure = list(map(float, row[2:]))
                self.X_data.append(feaure)

                # self.X_data.astype('float32')
                # self.Y_data.astype('float32')

                # data normalize
                # scaler = MinMaxScaler(feature_range=(0, 1))
                # self.X_data = scaler.fit_transform(self.X_data)

        # return (self.X_data, self.Y_data)

        """ 数据整形、规范化等 """
        ms = MinMaxScaler()
        self.X_data = ms.fit_transform(self.X_data)
        #时间序列填充
        length_of_sequence = len(self.X_data)

        for i in range(0, length_of_sequence - self.seq_length + 1, self.stride):
            self.X_.append(self.X_data[i: i + self.seq_length])
            self.Y_.append(self.Y_data[i: i + self.seq_length])

        self.X_ = np.array(self.X_).reshape(len(self.X_), self.seq_length, self.feature_length)
        self.Y_ = np.array(self.Y_).reshape(len(self.Y_), self.seq_length, 1)

        print(self.X_.shape)
        print(self.Y_.shape)

        """ 分为训练数据和验证数据 """
        train_len = int(len(self.X_) * 0.9)
        validation_len = len(self.X_) - train_len

        X_train, X_valid, Y_train, Y_valid = \
            train_test_split(self.X_, self.Y_, test_size=validation_len)

        return X_train, X_valid, Y_train, Y_valid
