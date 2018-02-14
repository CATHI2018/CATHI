'''
CATHI
'''

import tensorflow as tf

TEST_DATASET_PATH = '/CATHI/data/data_test.txt'
SAVE_DATA_DIR = '/CATHI/tf_data_traj/'

tf.app.flags.DEFINE_string('data_dir', SAVE_DATA_DIR + 'data', 'Data directory')
tf.app.flags.DEFINE_string('model_dir', SAVE_DATA_DIR + 'nn_models', 'Train directory')
tf.app.flags.DEFINE_string('results_dir', SAVE_DATA_DIR + 'results', 'Train directory')

tf.app.flags.DEFINE_float('learning_rate', 0.5, 'Learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.99, 'Learning rate decays by this much.')
tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'Clip gradients to this norm.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Batch size to use during training.')

tf.app.flags.DEFINE_integer('vocab_size', 100, 'Dialog vocabulary size.')
tf.app.flags.DEFINE_integer('size', 128, 'Size of each model layer.')
tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in the model.')

tf.app.flags.DEFINE_integer('max_train_data_size', 0, 'Limit on the size of training data (0: no limit).')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 100, 'How many training steps to do per checkpoint.')

FLAGS = tf.app.flags.FLAGS

BUCKETS = [(5, 10)]


