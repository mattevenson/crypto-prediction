import tensorflow as tf

class BaseEval:
    def __init__(self, sess, model, data, config):
        self.model = model
        self.config = config
        self.sess = sess
        self.data = data

    def evaluate(self):
        raise NotImplementedError
