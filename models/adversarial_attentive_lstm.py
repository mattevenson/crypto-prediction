from base.base_model import BaseModel
import tensorflow as tf

class AdvLSTMModel(BaseModel):
    def __init__(self, config):
        super(AdvLSTMModel, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.config.sequence_length, self.config.num_inputs])
        self.Y = tf.placeholder(tf.float32, [None, self.config.num_outputs])

        # x = tf.unstack(self.X, self.config.sequence_length, axis=1)

        m = tf.layers.dense(self.X, self.config.E, activation=tf.math.tanh)

        m_unstacked = tf.unstack(m, self.config.sequence_length, axis=1)

        lstm = tf.contrib.rnn.LSTMBlockCell(self.config.hidden_units)

        h, c = tf.contrib.rnn.static_rnn(lstm, m_unstacked, dtype=tf.float32)

        u = tf.get_variable('u', shape=(self.config.E_prime, 1), dtype=tf.float32)

        alpha_tilde = tf.matmul(tf.transpose(u), tf.layers.dense(h[-1], self.config.E_prime, activation=tf.math.tanh))

        a = tf.reduce_sum(tf.multiply(h, tf.nn.softmax(alpha_tilde)))

        e = tf.concat(a, h[-1][-1])

        self.prediction = tf.layers.dense(e, self.config.num_outputs)

        with tf.name_scope("loss"):
            self.l2_regularization = self.config.lambda_l2_reg * sum(tf.nn.l2_loss(v) for v in tf.trainable_variables() if not ("Bias" or "bias" in v.name))
            self.mean_squared_error = tf.losses.mean_squared_error(self.Y, self.prediction)
            # self.loss = tf.losses.log_loss(self.Y, self.prediction)
            self.loss = self.mean_squared_error + self.l2_regularization

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss, global_step=self.global_step_tensor)


    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
