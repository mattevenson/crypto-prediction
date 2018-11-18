from base.base_eval import BaseEval
from tqdm import tqdm

class Evaler(BaseEval):
    def __init__(self, sess, model, data, config):
        super(Evaler, self).__init__(sess, model, data, config)

    def evaluate(self):
        train_loss, train_prediction = self.sess.run([self.model.mean_squared_error, self.model.prediction], feed_dict={self.model.X: self.data.X_train, self.model.Y: self.data.Y_train})
        test_loss, test_prediction = self.sess.run([self.model.mean_squared_error, self.model.prediction], feed_dict={self.model.X: self.data.X_test, self.model.Y: self.data.Y_test})

        train_prediction = self.data.scaler.inverse_transform(train_prediction)
        test_prediction = self.data.scaler.inverse_transform(test_prediction)

        return train_loss, train_prediction, test_loss, test_prediction
