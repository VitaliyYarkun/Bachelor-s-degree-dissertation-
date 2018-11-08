import tensorflow as tf
import argparse
import os
import sys
import itertools
from glob import glob
import numpy as np


tf_dataset = tf.data.Dataset
tf_iterator = tf.data.Iterator

NUM_CLASSES = 62

def input_parser(img_path, label):
    # read the img from file
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)

    return img_decoded, label

train_files = list(sorted(glob('Train_dataset/*/*')))
train_lab   = np.repeat(np.arange(NUM_CLASSES), 500)
val_files   = list(sorted(glob('Validation_dataset/*/*')))
val_lab     = np.repeat(np.arange(NUM_CLASSES), 100)

train_imgs = tf.constant(train_files)
train_labels = tf.constant(train_lab)

val_imgs = tf.constant(val_files)
val_labels = tf.constant(val_lab)

# create TensorFlow Dataset objects
tr_data = tf_dataset.from_tensor_slices((train_imgs, train_labels))
val_data = tf_dataset.from_tensor_slices((val_imgs, val_labels))

tr_data = tr_data.shuffle(len(train_files)).map(input_parser).repeat().batch(64)
val_data = val_data.shuffle(len(val_files)).map(input_parser).repeat().batch(128)

# create TensorFlow Iterator object
iterator = tf_iterator.from_structure(tr_data.output_types,
                                   tr_data.output_shapes)
next_element = iterator.get_next()

# create two initialization ops to switch between the datasets
training_init_op = iterator.make_initializer(tr_data)
validation_init_op = iterator.make_initializer(val_data)

training_dataset = []
validation_dataset = []

FLAGS = None

def train():
    sess = tf.InteractiveSession()

    with tf.name_scope('input'):
        x, y_ = next_element
        x = (tf.cast(x, tf.float32) - 128.0)/128.0
        y_ = tf.cast(y_, tf.int64)

    with tf.name_scope('input_shape'):
        image_shaped_input = tf.reshape(x, [-1, 128, 128, 3])
        tf.summary.image('input', image_shaped_input, NUM_CLASSES)

    # Length of filters specifies the number of conv layers and number of filters in each
    filters = [16, 32, 64]
    strides = [2, 2, 2]
    activations = [tf.nn.relu,]

    h = image_shaped_input

    for f, s, a in zip(filters,
                       itertools.cycle(strides),
                       itertools.cycle(activations)):

        h = tf.layers.conv2d(h,
                             filters=f,
                             kernel_size=3,
                             strides=s,
                             activation=a)
    
    h = tf.layers.flatten(h)
    y = tf.layers.dense(h, NUM_CLASSES)

    with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                    labels=y_, logits=y)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
                cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    tf.global_variables_initializer().run()

    for i in range(FLAGS.max_steps):
        if i % 10 == 0:
            sess.run(validation_init_op)
            summary, acc = sess.run([merged, accuracy])
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
            sess.run(training_init_op)
        else:
            if i % 100 == 99:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:
                summary, _ = sess.run([merged, train_step])
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False, help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout.')
    parser.add_argument(
            '--data_dir',
            type=str,
            default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                 'tensorflow/mnist/input_data'),
            help='Directory for storing input data')
    parser.add_argument(
            '--log_dir',
            type=str,
            default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                 'tensorflow/mnist/logs/mnist_with_summaries'),
            help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
