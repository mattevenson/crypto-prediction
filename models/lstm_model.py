from base.base_model import BaseModel
import tensorflow as tf

class LSTMModel(BaseModel):
    def __init__(self, config):
        super(LSTMModel, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.config.sequence_length, self.config.num_inputs])
        self.Y = tf.placeholder(tf.float32, [None, self.config.num_outputs])

        self.x = tf.unstack(self.X, self.config.sequence_length, axis=1)

        lstm_cells = []

        for u in self.config.rnn_units:
            lstm_cell = tf.contrib.rnn.LSTMBlockCell(u)
            lstm_cells.append(lstm_cell)

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)

        rnn_outputs, rnn_states = tf.contrib.rnn.static_rnn(stacked_lstm, self.x, dtype=tf.float32)

        dense1 = tf.layers.dense(rnn_outputs[-1], 20, activation=tf.nn.relu)
        self.prediction = tf.layers.dense(rnn_outputs[-1], self.config.num_outputs)
        # self.prediction = tf.layers.dense(dense1, self.config.num_outputs, activation=tf.nn.sigmoid)

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
