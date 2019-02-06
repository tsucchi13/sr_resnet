import os, sys
import cv2
import sklearn
import numpy as np
from itertools import chain
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle

class Preprocessor(object):
    def __init__(self, data, label, eyes = "both"):
        self.__train_data = None
        self.__test_data = None
        self.__train_label = None
        self.__tests_label = None
        self.__data_size = data[0][0][0].size
        self.__eyes = eyes
        self.__data = data
        self.__label = label
        self.__users = len(data)

    def split_data(self, n, stds=True, mms=False):
        train_idx = [i for i in range(len(self.__data))]
        test_idx = train_idx.pop(n)
        self.__train_label = [self.__label[i] for i in train_idx]
        self.__train_label = list(chain.from_iterable(self.__train_label))
        self.__train_label = np.array(self.__train_label).reshape(-1,1)
        self.__test_label = self.__label[test_idx]
        self.__test_label = np.array(self.__test_label).reshape(-1,1)


        if self.__eyes == "both":
            self.__train_data = [self.__data[i] for i in train_idx]
            test_data = self.__data[test_idx]
            if n==0:
                with open("test_data_0202_img_v2.pkl", "wb") as f:
                    pickle.dump([test_data , self.__test_label], f)
            self.__train_data = list(chain.from_iterable(self.__train_data))
            self.__train_data = np.array(self.__train_data)
            self.__train_data = self.__train_data.reshape(-1, self.__data_size*2).astype(np.float64)
            self.__test_data = np.array(test_data).reshape(-1,
                            self.__data_size*2
                            ).astype(np.float64)

        elif self.__eyes == "left" or self.__eyes=="right":
            eye_idx = 0 if self.__eyes == "left" else 1
            self.__train_data = [self.__data[i] for i in train_idx]
            test_data = self.__data[test_idx]
            self.__train_data = list(chain.from_iterable(self.__train_data))
            self.__train_data = np.array(self.__train_data)[:,eye_idx]
            self.__train_data = self.__train_data.reshape(-1, self.__data_size).astype(np.float64)
            self.__test_data = np.array(test_data)[:,eye_idx].reshape(-1,
                            self.__data_size
                            ).astype(np.float64)

        self.minmax_scaler() if mms else None
        self.strandard_scaler() if stds else None


    def strandard_scaler(self):
        scaler = StandardScaler()
        scaler.fit(self.__train_data)
        self.__train_data = scaler.transform(self.__train_data)
        self.__test_data = scaler.transform(self.__test_data)

    def minmax_scaler(self):
        scaler = MinMaxScaler()
        scaler.fit(self.__train_data)
        self.__train_data = scaler.transform(self.__train_data)
        self.__test_data = scaler.transform(self.__test_data)
        # self.__train_data = self.__train_data/255
        # self.__test_data = self.__test_data/255


    @property
    def users(self):
        return self.__users

    @property
    def train_data(self):
        return self.__train_data

    @property
    def test_data(self):
        return self.__test_data

    @property
    def train_label(self):
        return self.__train_label

    @property
    def test_label(self):
        return self.__test_label
