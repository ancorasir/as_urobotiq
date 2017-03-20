"""Cross Entropy Method to infer the best grasp direction for date collection.
Not yet finished. For demonstration of CEM method.

By Fang Wan
"""
import googleGrasp as gg
import tensorflow as tf
import numpy as np
import googleGrasp_input as ggIn

def CEM():
    with tf.Graph().as_default():
        train_files = '/home/ancora-sirlab/grasp_dataset/grasping/grasping_dataset_058/grasping_dataset_058.tfrecord-00001-of-00068'
        train_files = tf.train.match_filenames_once(train_files)
        images, _, _ = ggIn.inputs(train_files, batch_size=32, num_epochs=1)
        motions = tf.placeholder(tf.float32, [64, 5], name='motions')
        is_training = tf.placeholder(tf.bool, name='is_training')
        inference = gg.inference(images, motions, is_training)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(init_op)
        # restore the trained network to calculate inference
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                print('start cem')
                # start the sampling from Gaussian distribution
                mean = [0, 0, 0, 0, 0]
                cov = np.diagflat([1, 1, 1, 1, 1])
                iteration = 0
                # google's work uses 3 iteration of optimization
                while iteration < 3:
                    # sampling 64 grasp directions vt, shape = [64,5]
                    Xs = []
                    N = 0
                    while N < 64:
                        X = np.random.multivariate_normal(mean, cov, 1)
                        if True:
                        # make sure the sample grasp in within the workspace of the robotic gripper
                        #if (rotations<=180) and (gripper in workspace):
                            Xs.append(X[0])
                            N += 1
                    Xs = np.array(Xs, dtype=np.float32)

                    # select the 6 best grasp directions by inferring to the network
                    # get images input from camera
                    # load the trained network, ??need more work
                    # performance = np.sum(Xs, axis=1) # for demonstration

                    performance = inference.eval(session=sess, feed_dict={
                        motions: Xs,
                        is_training: False
                        }).T[0]

                    # Sort X by objective function values (in ascending order)
                    best_idx = np.argsort(performance)[-6:]
                    best_Xs = np.array(Xs)[best_idx,:]

                    # Update parameters of distribution from the 6 best grasp directions
                    mean = np.mean(best_Xs, axis=0)
                    cov = np.cov(best_Xs, rowvar=0)
                    iteration += 1
                # use the optimized parameter to infer the best grasp direction
                Xs = []
                N = 0
                while N < 64:
                    X = np.random.multivariate_normal(mean, cov, 1)
                    if True:
                    #if (rotations<=180) and (gripper in workspace):
                        Xs.append(X[0])
                        N += 1
                Xs = np.array(Xs)
                # selecting the best grasp directions
                performance = np.sum(Xs, axis=1)

                best_idx = np.argsort(performance)[-1:]
                # just for test
                print('CEM batch end, results:')
                print(Xs[best_idx], performance[best_idx])
        except tf.errors.OutOfRangeError:
            print('queue out of range')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

def main(_):
    CEM()

if __name__ == '__main__':
    tf.app.run()
