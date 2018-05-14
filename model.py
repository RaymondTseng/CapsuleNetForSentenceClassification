
import tensorflow as tf
import data_helper
import utils

epsilon = 1e-9
iter_routing = 3
mask_with_y = True



# CapsuleNet for sentence classification
class CapsuleNet:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.build_network()
            self.loss()
            self.summary()

    def build_network(self):
        self.word2idx, word_embedding = data_helper.load_embedding(self.FLAGS.embedding_path)

        self.inputs, self.labels, self.train_num = utils.get_batch_data(self.FLAGS.train_data_path,
                                                        self.FLAGS.batch_size, self.FLAGS.seq_length, self.word2idx)
        self.labels = tf.one_hot(self.labels, depth=self.FLAGS.class_num)

        # embedding layer
        with tf.variable_scope('embedding'):
            embedding = tf.get_variable('embedding', shape=word_embedding.shape, dtype=tf.float32,
                                        initializer=tf.constant_initializer(word_embedding), trainable=False)
            # [batch, 30, 300]
            inputs = tf.nn.embedding_lookup(embedding, self.inputs)

            self.origin = tf.reduce_mean(inputs, axis=1)

            inputs = tf.expand_dims(inputs, -1)

        with tf.variable_scope('conv1_layer'):
            # [batch, 30, 1, 300]
            input_conv = tf.layers.conv2d(inputs, self.FLAGS.filter_num,
                                          [self.FLAGS.filter_size, self.FLAGS.embedding_size])

            # Primary Capsules layer
            with tf.variable_scope('PrimaryCaps_layer'):
                primaryCaps = CapsLayer(num_outputs=30, vec_len=10, with_routing=False, layer_type='CONV')

                caps1 = primaryCaps(input_conv, kernel_size=3, stride=1)


        # DigitCaps layer, return [batch_size, 2, vec_len, 1]
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(num_outputs=self.FLAGS.class_num, vec_len=32, with_routing=True, layer_type='FC')
            # [batch_size, 2, 100, 1]
            self.caps2 = digitCaps(caps1)

        # Decoder structure
        # 1. Do masking, how:
        with tf.variable_scope('Masking'):
            # [batch_size, class_num, vec_len, 1] => [batch_size, class_num, 1, 1]
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2),
                                               axis=2, keep_dims=True) + epsilon)
            # [batch_size, class_num, 1, 1]
            self.softmax_v = tf.nn.softmax(self.v_length, dim=1)


            # [batch_size, class_num, 1, 1] => [batch_size] (index)
            self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
            self.argmax_idx = tf.reshape(self.argmax_idx, shape=[-1])

            # Method 1.
            if not mask_with_y:
                # indexing

                # [batch_size, class_num, 1, 1]
                one_hot_idx = tf.expand_dims(tf.expand_dims(tf.one_hot(self.argmax_idx, self.FLAGS.class_num), -1), -1)
                # [batch_size, 1, vec_len, 1]
                self.masked_v = tf.reduce_sum(self.caps2 * one_hot_idx, 1)

            # Method 2. masking with true label, default mode
            else:

                self.masked_v = tf.multiply(tf.squeeze(self.caps2, 3),
                                            tf.reshape(self.labels, (-1, self.FLAGS.class_num, 1)))
                # [batch_size, class_num, 1]
                self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2) + epsilon)

        # 2. Reconstructe the data with 3 FC layers
        # [batch_size, 2, vec_len, 1] => [batch_size, 64] => [batch_size, 300]
        with tf.variable_scope('Decoder'):
            vector_j = tf.reshape(self.masked_v, shape=(-1, self.FLAGS.class_num * 32))
            fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=256)

            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=512)

            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=300, activation_fn=tf.tanh)


    def loss(self):
        # 1. The margin loss

        v_length = tf.squeeze(self.v_length, axis=2)

        self.origin_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=v_length))


        # 2. The reconstruction loss
        squared = tf.square(self.decoded - self.origin)
        self.reconstruction_err = tf.reduce_mean(squared)

        # 3. Total loss
        self.total_loss = self.origin_loss + self.FLAGS.regularization_scale * self.reconstruction_err

        self.optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate).minimize(self.total_loss)

    def summary(self):
        correct_prediction = tf.equal(tf.to_int32(tf.argmax(self.labels, axis=1)), self.argmax_idx)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

class CapsLayer:

    def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type

    def __call__(self, input, kernel_size=None, stride=None, reuse=None):

        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            # the PrimaryCaps layer, a convolutional layer
            # input = [batch, 30, 1, 300]
            input = tf.transpose(input, [0, 1, 3, 2])
            if not self.with_routing:
                # [batch, 29, 1, 300]
                capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.vec_len,
                                                    [self.kernel_size, self.num_outputs * self.vec_len],
                                                    self.stride, padding="VALID",
                                                    activation_fn=tf.nn.relu, reuse=reuse)
                capsules = tf.reshape(capsules, [-1, 840, self.vec_len, 1])

                capsules = squash(capsules)
                return capsules

        if self.layer_type == 'FC':
            if self.with_routing:
                # the DigitCaps layer, a fully connected layer
                # Reshape the input into [batch_size, 840, 1, 10, 1]
                input = tf.reshape(input, [-1, 840, 1, 10, 1])

                with tf.name_scope('routing'):
                    # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                    shape_tensor = tf.reduce_sum(input, axis=-2, keep_dims=True)
                    b_IJ = tf.tile(tf.zeros_like(shape_tensor, dtype=tf.float32), [1, 1, self.num_outputs, 1, 1])

                    capsules = self.routing(input, b_IJ)
                    capsules = tf.squeeze(capsules, axis=1)

            return capsules

    def routing(self, input, b_IJ):
        # input: [batch_size, 840, 1, 10, 1]
        # b_IJ: [batch_size, 840, 2, 1, 1]
        # [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1]
        W = tf.get_variable('weight', shape=(1, 840, self.num_outputs * self.vec_len, 10, 1), dtype=tf.float32,
                            initializer=tf.random_normal_initializer)
        biases = tf.get_variable('bias', shape=(1, 1, self.num_outputs, self.vec_len, 1))


        input = tf.tile(input, [1, 1, self.num_outputs * self.vec_len, 1, 1])
        # [1, 840, num_outputs * vec_len, 10, 1] * [batch_size, 840, 1, 10, 1]
        # ==> [batch_size, 840, num_outputs * vec_len, 1, 1]
        u_hat = tf.reduce_sum(W * input, axis=3, keep_dims=True)
        u_hat = tf.reshape(u_hat, shape=[-1, 840, self.num_outputs, self.vec_len, 1])

        # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
        u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

        # line 3,for r iterations do
        for r_iter in range(iter_routing):
            with tf.variable_scope('iter_' + str(r_iter)):
                # [batch_size, 840, num_outputs, 1, 1]
                c_IJ = tf.nn.softmax(b_IJ, dim=2)

                if r_iter == iter_routing - 1:
                    # [batch_size, 840, num_outputs, vec_len, 1]
                    s_J = tf.multiply(c_IJ, u_hat)
                    # [batch_size, 1, num_outputs, vec_len, 1]
                    s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True) + biases
                    # [batch_size, 1, num_outputs, vec_len, 1]
                    v_J = squash(s_J)
                elif r_iter < iter_routing - 1:
                    # [batch_size, 840, num_outputs, 1, 1] * [batch_size, 840, num_outputs, vec_len, 1]
                    # ==> [batch_size, 840, num_outputs, vec_len, 1]
                    s_J = tf.multiply(c_IJ, u_hat_stopped)
                    # [batch_size, 1, num_outputs, vec_len, 1]
                    s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True) + biases
                    v_J = squash(s_J)


                    # [batch_size, 840, num_outputs, vec_len, 1]
                    v_J_tiled = tf.tile(v_J, [1, 840, 1, 1, 1])
                    # [batch_size, 840, num_outputs, 1, 1]
                    u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keep_dims=True)

                    b_IJ += u_produce_v

        return v_J

def squash(vector):

    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)
