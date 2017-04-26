from utils import get_data_from_location
from graph import generategraph
from model import Model
from settings import *
import tensorflow as tf
import numpy as np
import time

dataset, samplerate = get_data_from_location(FLAGS.data_dir)

# this is for neural network
model = Model(FLAGS.learning_rate, FLAGS.num_layers, FLAGS.sound_batch_size, FLAGS.size)

# start the session graph
sess = tf.InteractiveSession()
    
# initialize global variables
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(tf.global_variables())

def train():

    print "Train for " + str(FLAGS.max_epoch) + " iteration"

    for i in xrange(FLAGS.max_epoch):
        last_time = time.time()
        init_value = np.zeros((1, FLAGS.num_layers * 2 * FLAGS.size))
    
        for x in xrange(0, dataset.shape[0], FLAGS.sound_batch_size):
        
            batch = np.zeros((1, 1, FLAGS.sound_batch_size))
            batch_y = np.zeros((1, 1, FLAGS.sound_batch_size))
    
            batch[0, 0, :] = dataset[x: x + FLAGS.sound_batch_size]
            batch_y[0, 0, :] = dataset[x + FLAGS.sound_batch_size: x + (2 * FLAGS.sound_batch_size)]
        
            _, loss = sess.run([model.optimizer, model.cost], feed_dict={model.X: batch, model.Y: batch_y,
                                                                         model.back_hidden_layer: init_value, model.forward_hidden_layer: init_value,})

        diff = time.time() - last_time
        print "batch: " + str(i + 1) + ", loss: " + str(np.mean(loss)) + ", speed: " + str(diff) + " s / epoch"
    
        if (i + 1) % FLAGS.checkpoint_epoch == 0:
            saver.save(sess, FLAGS.train_dir + "model.ckpt")
            generategraph(dataset, model, sess)

def main():
    if FLAGS.decode:
        test()
    else:
        train()
        
main()