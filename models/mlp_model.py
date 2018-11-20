from base.base_model import BaseModel
import tensorflow as tf

class MLPModel(BaseModel):
    def __init__(self, config):
        super(MLPModel, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.config.sequence_length, self.config.num_inputs])
        self.Y = tf.placeholder(tf.float32, [None, self.config.num_outputs])

        self.x = tf.contrib.layers.flatten(self.X)

        dense1 = tf.layers.dense(self.x, 20, activation=tf.nn.relu)
        dense2 = tf.layers.dense(dense1, 20, activation=tf.nn.relu)
        self.prediction = tf.layers.dense(dense2, self.config.num_outputs)

        with tf.name_scope("loss"):
            self.mean_squared_error = tf.losses.mean_squared_error(self.Y, self.prediction)
            self.loss = self.mean_squared_error

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss, global_step=self.global_step_tensor)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
