'''
Reading grasp data in TFrecords format for training.
reference resources:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/

By Fang Wan
'''

import tensorflow as tf
import numpy as np

def read_and_decode_single_example(filename_queue):
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of all fields.
            'gripper/status': tf.FixedLenFeature([], tf.float32),
            'grasp/image/encoded': tf.FixedLenFeature([], tf.string),
            'grasp/0/params': tf.FixedLenFeature([5], tf.float32),
            'grasp/0/image/encoded': tf.FixedLenFeature([], tf.string),
            'grasp/1/params': tf.FixedLenFeature([5], tf.float32),
            'grasp/1/image/encoded': tf.FixedLenFeature([], tf.string)
        })
    # calculate successful label
    cond = tf.greater(features['gripper/status'], 0.01)
    label = tf.select(cond, 1.0, 0.0)
    # calculate input images to the network
    grasp = tf.image.decode_jpeg(features['grasp/image/encoded'], channels=3)
    grasp_0 = tf.image.decode_jpeg(features['grasp/0/image/encoded'], channels=3)
    grasp_1 = tf.image.decode_jpeg(features['grasp/1/image/encoded'], channels=3)
    cropped_grasp = tf.random_crop(grasp, [472, 472, 3])
    cropped_grasp_0 = tf.random_crop(grasp_0, [472, 472, 3])
    cropped_grasp_1 = tf.random_crop(grasp_1, [472, 472, 3])
    images = tf.stack(
                      [tf.concat(0, [cropped_grasp,cropped_grasp_0]),
                       tf.concat(0, [cropped_grasp,cropped_grasp_1])
                       ], axis=0
                      )
    images = tf.cast(images, tf.float32) # [2,472*2,472,3]
    # calculate input motions to the network
    motions = tf.stack([features['grasp/0/params'], features['grasp/1/params']], axis=0)
    # duplicate labels to the same size as images input
    labels = tf.tile(tf.reshape(label, [1,1]),
                     [images.get_shape()[0].value,1]
                     ) #[2,1]
    labels = tf.cast(labels, tf.float32)
    return images, motions, labels

def inputs(filenames, batch_size, num_epochs):
    """Reads input data num_epochs times.

    Args:
        filename: a list of all file names which are used for training.
        batch_size: Number of examples per batch.
        num_epochs: Number of times to read the input data, or 0/None to train forever.

    Returns:
        A tuple (images, labels), where:
        * images is a float tensor with shape [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3].
        * labels is a float tensor with shape [batch_size] with the true label.
        Note that an tf.train.QueueRunner is added to the graph, which
        must be run using e.g. tf.train.start_queue_runners().
    """
    if not num_epochs: num_epochs = None
    with tf.name_scope('input'):
        # creates a FIFO queue for holding the filenames until the reader needs them. string_input_producer has options for shuffling and setting a maximum number of epochs. A queue runner adds the whole list of filenames to the queue once for each epoch. We grab a filename off our queue of filenames and use it to get examples from a TFRecordReader. Both the queue and the TFRecordReader have some state to keep track of where they are.
        # On initialization filename queue is empty. This is where the concept of QueueRunners comes in. It is simply a thread that uses a session and calls an enqueue op over and over again.
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
        images, motions, labels = read_and_decode_single_example(filename_queue)
        # groups examples into batches randomly
        # shuffle_batch constructs a RandomShuffleQueue and proceeds to fill it with individual image and labels. This filling is done on a separate thread with a QueueRunner. The RandomShuffleQueue accumulates examples sequentially until it contains batch_size +min_after_dequeue examples are present. It then selects batch_size random elements from the queue to return.
        images_batch, motions_batch, labels_batch = tf.train.shuffle_batch([images, motions, labels], batch_size=batch_size, capacity=1000 + 3 * batch_size, min_after_dequeue=1000)
        actual_batch_size = images_batch.get_shape()[0].value*images_batch.get_shape()[1].value
        return tf.reshape(images_batch, [actual_batch_size, 472*2, 472, 3]), tf.reshape(motions_batch, [actual_batch_size, 5]), tf.reshape(labels_batch, [actual_batch_size, 1])
