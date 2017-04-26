import tensorflow as tf

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_integer("sound_batch_size", 1000, "Sound batch size to use during training.")
tf.app.flags.DEFINE_integer("checkpoint_epoch", 100, "Checkpoint to save and testing")
tf.app.flags.DEFINE_integer("max_epoch", 1000, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("size", 32, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_string("train_dir", "/home/huseinzol05/AI/music/", "Training directory.")
tf.app.flags.DEFINE_string("data_dir", "/home/huseinzol05/AI/sound/data/", "data location.")

tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")

memory_duringtraining = 0.8
memory_duringtesting = 0.1

FLAGS = tf.app.flags.FLAGS

def get_data_type():
    return tf.float64 if FLAGS.use_fp64 else tf.float32

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
    
if FLAGS.decode:
    config.gpu_options.per_process_gpu_memory_fraction = memory_duringtesting
else:
    config.gpu_options.per_process_gpu_memory_fraction = memory_duringtraining