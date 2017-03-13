"""Cross Entropy Method to infer the best grasp direction for date collection.
Not yet finished. For demonstration of CEM method.

By Fang Wan
"""
import googleGrasp as gg
import tensorflow as tf
import numpy as np

def CEM():
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
        Xs = np.array(Xs)
        # selecting the 6 best grasp directions by inferring to the network
        # performance = np.sum(Xs, axis=1) # for demonstration
        images_batch, motions_batch,
        is_training = tf.placeholder(tf.bool, name='is_training')
        performance = gg.inference(images_batch, motions_batch, is_training)
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
    # performance = conv_net(s, v, W, b)
    # with tf.Session() as sess:
    #    sess.run(init)
    #    sess.run(performance, feed_dict={s: result, v: Xs})
    best_idx = np.argsort(performance)[-1:]
    return Xs[best_idx], performance[best_idx]

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, outputfie)
    sess.run(performance, feed_dict={s: result, v: Xs})
