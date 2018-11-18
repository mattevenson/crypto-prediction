from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np

class Trainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(Trainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.data.num_iter_per_epoch))

        for _ in loop:
            self.train_step()

        train_loss = self.sess.run([self.model.mean_squared_error], feed_dict={self.model.X: self.data.X_train, self.model.Y: self.data.Y_train})
        test_loss = self.sess.run([self.model.mean_squared_error], feed_dict={self.model.X: self.data.X_test, self.model.Y: self.data.Y_test})

        cur_it = self.model.global_step_tensor.eval(self.sess)

        summaries_dict = {
            'train_loss': np.mean(train_loss),
            'test_loss': np.mean(test_loss)
        }

        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
       batch_X, batch_Y =  next(self.data.next_batch())
       self.sess.run([self.model.train_step], feed_dict={self.model.X: batch_X, self.model.Y: batch_Y})
