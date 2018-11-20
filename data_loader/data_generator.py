import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import math
import numpy as np

class DataGenerator:
    def __init__(self, config, features, labels):
        self.config = config

        self.features_train, self.features_test, self.labels_train, self.labels_test = train_test_split(features, labels, test_size=self.config.test_size, shuffle=False)

        self.features_scaler = MinMaxScaler()
        self.features_scaler.fit(self.features_train)
        self.labels_scaler = MinMaxScaler()
        self.labels_scaler.fit(self.labels_train)

        self.X_train, self.Y_train = self.sliding_window(self.features_scaler.transform(self.features_train), self.labels_scaler.transform(self.labels_train), self.config.sequence_length)
        self.X_test, self.Y_test = self.sliding_window(self.features_scaler.transform(self.features_test), self.labels_scaler.transform(self.labels_test), self.config.sequence_length)

        self.num_iter_per_epoch = math.ceil(len(self.X_train) / self.config.batch_size)

    def sliding_window(self, features, labels, sequence_length, step=1):
        X = []
        Y = []

        for i in range(0, len(features) - sequence_length, step):
            X.append(features[i:i + sequence_length])
            Y.append(labels[i + sequence_length])

        X = np.array(X)
        Y = np.array(Y)

        return X, Y

    def next_batch(self):
        p = np.random.permutation(len(self.X_train))
        self.X_train_shuffled = self.X_train[p]
        self.Y_train_shuffled = self.Y_train[p]
        for i in range(0, len(self.X_train_shuffled) - self.config.batch_size, self.config.batch_size):
            end = min(i + self.config.batch_size, len(self.X_train_shuffled))
            yield self.X_train_shuffled[i:end], self.Y_train_shuffled[i:end]
