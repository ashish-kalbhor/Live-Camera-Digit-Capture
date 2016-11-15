from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
import six.moves.cPickle as pickle
from six.moves import range
import tensorflow as tf

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None


def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 1% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

## Downloading the DataSet
train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

num_classes = 10
np.random.seed(133)


def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders

## Extracting the DataSet
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

## Load the data and create Train and Test dataSets.
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

## To balance the data, Merge and Prune the training Data.
def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
    train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)


## Save the complete data in the end.
pickle_file = 'notMNIST.pickle'

# try:
#   f = open(pickle_file, 'wb')
#   save = {
#     'train_dataset': train_dataset,
#     'train_labels': train_labels,
#     'valid_dataset': valid_dataset,
#     'valid_labels': valid_labels,
#     'test_dataset': test_dataset,
#     'test_labels': test_labels,
#     }
#   pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
#   f.close()
# except Exception as e:
#   print('Unable to save data to', pickle_file, ':', e)
#   raise

# statinfo = os.stat(pickle_file)
# print('Compressed pickle size:', statinfo.st_size)

######################Building Logistic Model#############################
# (samples, width, height) = train_dataset.shape
# X_train = np.reshape(train_dataset, (samples, width*height))[0:5000]
#
# model = LogisticRegression()
# model = model.fit(X_train, train_labels[0:5000])
#
# print(model.score(X_train, train_labels[0:5000]))
##########################################################################
######################Fully Connected Network############################
#print('######################Fully Connected Network############################')
#pickle_file = 'notMNIST.pickle'

# with open(pickle_file, 'rb') as f:
#   save = pickle.load(f)
#   train_dataset = save['train_dataset']
#   train_labels = save['train_labels']
#   valid_dataset = save['valid_dataset']
#   valid_labels = save['valid_labels']
#   test_dataset = save['test_dataset']
#   test_labels = save['test_labels']
#   del save  # hint to help gc free up memory
#   print('Training set', train_dataset.shape, train_labels.shape)
#   print('Validation set', valid_dataset.shape, valid_labels.shape)
#   print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
# train_subset = 10000
#
# graph = tf.Graph()
# with graph.as_default():
#     # Input data.
#     # Load the training, validation and test data into constants that are
#     # attached to the graph.
#     tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
#     tf_train_labels = tf.constant(train_labels[:train_subset])
#     tf_valid_dataset = tf.constant(valid_dataset)
#     tf_test_dataset = tf.constant(test_dataset)
#
#     # Variables.
#     # These are the parameters that we are going to be training. The weight
#     # matrix will be initialized using random values following a (truncated)
#     # normal distribution. The biases get initialized to zero.
#     weights = tf.Variable(
#         tf.truncated_normal([image_size * image_size, num_labels]))
#     biases = tf.Variable(tf.zeros([num_labels]))
#
#     # Training computation.
#     # We multiply the inputs with the weight matrix, and add biases. We compute
#     # the softmax and cross-entropy (it's one operation in TensorFlow, because
#     # it's very common, and it can be optimized). We take the average of this
#     # cross-entropy across all training examples: that's our loss.
#     logits = tf.matmul(tf_train_dataset, weights) + biases
#     loss = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
#
#     # Optimizer.
#     # We are going to find the minimum of this loss using gradient descent.
#     optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#
#     # Predictions for the training, validation, and test data.
#     # These are not part of training, but merely here so that we can report
#     # accuracy figures as we train.
#     train_prediction = tf.nn.softmax(logits)
#     valid_prediction = tf.nn.softmax(
#         tf.matmul(tf_valid_dataset, weights) + biases)
#     test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
#
# num_steps = 801
#
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
#
# with tf.Session(graph=graph) as session:
#   # This is a one-time operation which ensures the parameters get initialized as
#   # we described in the graph: random weights for the matrix, zeros for the
#   # biases.
#   tf.initialize_all_variables().run()
#   print('Initialized')
#   for step in range(num_steps):
#     # Run the computations. We tell .run() that we want to run the optimizer,
#     # and get the loss value and the training predictions returned as numpy
#     # arrays.
#     _, l, predictions = session.run([optimizer, loss, train_prediction])
#     if (step % 100 == 0):
#       print('Loss at step %d: %f' % (step, l))
#       print('Training accuracy: %.1f%%' % accuracy(
#         predictions, train_labels[:train_subset, :]))
#       # Calling .eval() on valid_prediction is basically like calling run(), but
#       # just to get that one numpy array. Note that it recomputes all its graph
#       # dependencies.
#       print('Validation accuracy: %.1f%%' % accuracy(
#         valid_prediction.eval(), valid_labels))
#   print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


# Switch to stochastic gradient descent training instead, which is much faster.
# batch_size = 128
# n_hidden_nodes = 1024
#
# graph = tf.Graph()
# with graph.as_default():
#     # Input data. For the training data, we use a placeholder that will be fed
#     # at run time with a training minibatch.
#     tf_train_dataset = tf.placeholder(tf.float32,
#                                       shape=(batch_size, image_size * image_size))
#     tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#     tf_valid_dataset = tf.constant(valid_dataset)
#     tf_test_dataset = tf.constant(test_dataset)
#
#     # Variables.
#     weights_01 = tf.Variable(
#         tf.truncated_normal([image_size * image_size, n_hidden_nodes]))
#     weights_12 = tf.Variable(
#         tf.truncated_normal([n_hidden_nodes, num_labels]))
#     biases_01 = tf.Variable(tf.zeros([n_hidden_nodes]))
#     biases_12 = tf.Variable(tf.zeros([num_labels]))
#
#     #--------------------------------------------------------
#     # Converting into 1 hidden layer neural network
#     z_01 = tf.matmul(tf_train_dataset, weights_01) + biases_01
#     h1 = tf.nn.relu(z_01)
#     z_12 = tf.matmul(h1, weights_12) + biases_12
#
#     loss = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(z_12, tf_train_labels))
#
#     # Optimizer.
#     optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#
#     # Predictions for the training, validation, and test data.
#     train_prediction = tf.nn.softmax(z_12)
#     valid_prediction = tf.nn.softmax(
#         tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_01) + biases_01), weights_12) + biases_12)
#     test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_01) + biases_01), weights_12) + biases_12)
#
# num_steps = 3001
#
# with tf.Session(graph=graph) as session:
#   tf.initialize_all_variables().run()
#   print("Initialized")
#   for step in range(num_steps):
#     # Pick an offset within the training data, which has been randomized.
#     # Note: we could use better randomization across epochs.
#     offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#     # Generate a minibatch.
#     batch_data = train_dataset[offset:(offset + batch_size), :]
#     batch_labels = train_labels[offset:(offset + batch_size), :]
#     # Prepare a dictionary telling the session where to feed the minibatch.
#     # The key of the dictionary is the placeholder node of the graph to be fed,
#     # and the value is the numpy array to feed to it.
#     feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
#     _, l, predictions = session.run(
#       [optimizer, loss, train_prediction], feed_dict=feed_dict)
#     if (step % 500 == 0):
#       print("Minibatch loss at step %d: %f" % (step, l))
#       print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
#       print("Validation accuracy: %.1f%%" % accuracy(
#         valid_prediction.eval(), valid_labels))
#   print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

###########Convulational Networks#################################

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal(
        [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))


    # Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases


    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

num_steps = 1001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
