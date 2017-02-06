'''
Examples of reading grasp data from google.
under folder /google-test/grasping_dataset_057/, run this script
'''
import tensorflow as tf
import os
from PIL import Image

current_path = os.getcwd()
input_file = os.path.join(current_path, "grasping_dataset_057.tfrecord")

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    i=0
    for serialized_example in tf.python_io.tf_record_iterator(input_file):
        # example contains features, features contains a map of strings to feature and feature contains one of a float_list, a bytes_list or a int64_list.
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        # reading and printing some of the parameters
        gripper_status = example.features.feature["gripper/status"].float_list.value
        grasp_0_params = example.features.feature["grasp/0/params"].float_list.value
        grasp_1_params = example.features.feature["grasp/1/params"].float_list.value
        print("gripper_status: {}".format(gripper_status))
        print("grasp_0_params: {}".format(grasp_0_params))
        print("grasp_1_params: {}".format(grasp_1_params))
        # reading and decoding some of the images
        image_grasp_0 = example.features.feature["grasp/0/image/encoded"].bytes_list.value
        image_grasp_1 = example.features.feature["grasp/1/image/encoded"].bytes_list.value
        image_grasp = example.features.feature["grasp/image/encoded"].bytes_list.value
        image_gripper = example.features.feature["gripper/image/encoded"].bytes_list.value
        image_post_drop = example.features.feature["post_drop/image/encoded"].bytes_list.value
        image_post_grasp = example.features.feature["post_grasp/image/encoded"].bytes_list.value
        grasp_0 = tf.image.decode_jpeg(image_grasp_0[0], channels=3)
        grasp_1 = tf.image.decode_jpeg(image_grasp_1[0], channels=3)
        grasp = tf.image.decode_jpeg(image_grasp[0], channels=3)
        gripper = tf.image.decode_jpeg(image_gripper[0], channels=3)
        post_drop = tf.image.decode_jpeg(image_post_drop[0], channels=3)
        post_grasp = tf.image.decode_jpeg(image_post_grasp[0], channels=3)
        # save images
        img = Image.fromarray(sess.run(grasp_0), 'RGB')
        img.save('grasp_0_'+str(i)+'.png')
        img = Image.fromarray(sess.run(grasp_1), 'RGB')
        img.save('grasp_1_'+str(i)+'.png')
        img = Image.fromarray(sess.run(grasp), 'RGB')
        img.save('grasp_'+str(i)+'.png')
        img = Image.fromarray(sess.run(gripper), 'RGB')
        img.save('gripper_'+str(i)+'.png')
        img = Image.fromarray(sess.run(post_drop), 'RGB')
        img.save('post_drop_'+str(i)+'.png')
        img = Image.fromarray(sess.run(post_grasp), 'RGB')
        img.save('post_grasp_'+str(i)+'.png')
        i=i+1
