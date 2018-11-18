import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import math
import numpy as np

class DataGenerator:
    def __init__(self, config, data):
        self.config = config

        self.data_train, self.data_test = train_test_split(data, test_size=self.config.test_size, shuffle=False)

        self.scaler = MinMaxScaler()
        self.scaler.fit(self.data_train)

        self.X_train, self.Y_train = self.sliding_window(self.scaler.transform(self.data_train), self.config.sequence_length)
        self.X_test, self.Y_test = self.sliding_window(self.scaler.transform(self.data_test), self.config.sequence_length)

        self.num_iter_per_epoch = math.ceil(len(self.X_train) / self.config.batch_size)

    def sliding_window(self, data, sequence_length, step=1):
        X = []
        Y = []

        for i in range(0, len(data) - sequence_length, step):
            X.append(data[i:i + sequence_length])
            Y.append(data[i + sequence_length])

        X = np.array(X)
        Y = np.array(Y)

        return X, Y

    def next_batch(self):
        for i in range(0, len(self.X_train) - self.config.batch_size, self.config.batch_size):
            end = min(i + self.config.batch_size, len(self.X_train))
            yield self.X_train[i:end], self.Y_train[i:end]
