import tensorflow as tf
import numpy as np

class Model:
    
    def __init__(self, learning_rate, num_layers, shape_3d, size_layer):

        self.num_layers = num_layers
        self.size_layer = size_layer
        
        rnn_cells = tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple = False)
        
        self.back_rnn_cells =  tf.nn.rnn_cell.MultiRNNCell([rnn_cells] * num_layers, state_is_tuple = False)
        self.forward_rnn_cells =  tf.nn.rnn_cell.MultiRNNCell([rnn_cells] * num_layers, state_is_tuple = False)
        
        self.X = tf.placeholder(tf.float32, (None, None, shape_3d))
        self.Y = tf.placeholder(tf.float32, (None, None, shape_3d))
        
        self.net_last_state = np.zeros((num_layers * 2 * size_layer))
        
        self.back_hidden_layer = tf.placeholder(tf.float32, shape=(None, num_layers * 2 * size_layer))
        self.forward_hidden_layer = tf.placeholder(tf.float32, shape=(None, num_layers * 2 * size_layer))
        
        self.outputs, self.last_state = tf.nn.bidirectional_dynamic_rnn(self.forward_rnn_cells, self.back_rnn_cells, 
                                                                        self.X, sequence_length = [1], 
                                                                        initial_state_fw = self.forward_hidden_layer, initial_state_bw = self.back_hidden_layer,
                                                                        dtype = tf.float32)
        
        self.rnn_W = tf.Variable(tf.random_normal((size_layer, shape_3d)))
        self.rnn_B = tf.Variable(tf.random_normal((shape_3d,)))
            
        # linear dimension for x in (Wx + B)
        outputs_reshaped = tf.reshape(self.outputs, [-1, size_layer])
        
        y_batch_long = tf.reshape(self.Y, [-1, shape_3d])

        # y = Wx + B
        self.logits = (tf.matmul(outputs_reshaped, self.rnn_W) + self.rnn_B)
        
        self.cost = tf.abs(y_batch_long - self.logits)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

    def step(self, sess, x, init_zero_state = True):

        # Reset the initial state of the network.
        if init_zero_state:
            init_value = np.zeros((self.num_layers * 2 * self.size_layer,))
        else:
            init_value = self.net_last_state
        
        # we want to get the constant in our output layer, same size as dimension input
        probs, next_lstm_state = sess.run([self.logits, self.last_state], feed_dict={self.X:[x], self.hidden_layer:[init_value]})

        self.net_last_state = next_lstm_state[0]

        return probs