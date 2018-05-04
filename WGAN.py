import tensorflow as tf
import os
import re
from glob import glob
import numpy as np
import matplotlib.image as mpimg

class WGAN(object):
    def __init__(self, sess, learning_rate=0.001, num_epochs=100, batch_size=32, display_step=10, save_step=50,
                 n_inputs=4, n_filters=[16, 32, 64],
                 gen_dim=500, input_height=40, input_width=40, num_layers=4, size_filter=5, rgb_filters=3, strides=2,
                 path_save='./saved_model', path_data='./data'):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.display_step = display_step
        self.save_step = save_step
        self.n_inputs = [input_height, input_width, rgb_filters]
        self.n_filters = n_filters
        self.gen_dim = gen_dim
        self.size_filter = size_filter
        self.rgb_filters = rgb_filters
        self.strides = strides

        self.input_height = input_height
        self.input_width = input_width

        self.path_save = path_save
        self.path_data = path_data
        self.images = self.load_data()

        self.sess = sess

        self.num_layers = num_layers

        self.weights = {
            'd_w1': tf.Variable(
                tf.truncated_normal([self.size_filter, self.size_filter, self.rgb_filters, self.n_filters[0]],
                                    stddev=0.01)),
            'd_w2': tf.Variable(
                tf.truncated_normal([self.size_filter, self.size_filter, self.n_filters[0], self.n_filters[1]],
                                    stddev=0.01)),
            'd_w3': tf.Variable(
                tf.truncated_normal([self.size_filter, self.size_filter, self.n_filters[1], self.n_filters[2]],
                                    stddev=0.01)),
            'd_w4': tf.Variable(
                tf.truncated_normal([self.size_filter * self.size_filter * self.n_filters[2], 1], stddev=0.01)),

            'g_w1': tf.Variable(
                tf.truncated_normal([self.gen_dim, self.size_filter * self.size_filter * self.n_filters[2]],
                                    stddev=0.01)),
            'g_w2': tf.Variable(
                tf.truncated_normal([self.size_filter, self.size_filter, self.n_filters[1], self.n_filters[2]],
                                    stddev=0.01)),
            'g_w3': tf.Variable(
                tf.truncated_normal([self.size_filter, self.size_filter, self.n_filters[0], self.n_filters[1]],
                                    stddev=0.01)),
            'g_w4': tf.Variable(
                tf.truncated_normal([self.size_filter, self.size_filter, self.rgb_filters, self.n_filters[0]],
                                    stddev=0.01)),
        }

        self.biases = {
            'd_w1': tf.Variable(tf.truncated_normal([self.n_filters[0]], stddev=0.01)),
            'd_w2': tf.Variable(tf.truncated_normal([self.n_filters[1]], stddev=0.01)),
            'd_w3': tf.Variable(tf.truncated_normal([self.n_filters[2]], stddev=0.01)),
            'd_w4': tf.Variable(tf.truncated_normal([1], stddev=0.01)),

            'g_w1': tf.Variable(
                tf.truncated_normal([self.size_filter * self.size_filter * self.n_filters[2]], stddev=0.01)),
            'g_w2': tf.Variable(tf.truncated_normal([self.n_filters[1]], stddev=0.01)),
            'g_w3': tf.Variable(tf.truncated_normal([self.n_filters[0]], stddev=0.01)),
            'g_w4': tf.Variable(tf.truncated_normal([3], stddev=0.01))
        }

        self.X = tf.placeholder(tf.float32, [None] + self.n_inputs)
        self.Z = tf.placeholder(tf.float32, [None, self.gen_dim])

        # Construct discriminator and generator
        self.gen_sample = self.generator(self.Z)
        _, self.dis_real = self.discriminator(self.X)
        _, self.dis_fake = self.discriminator(self.gen_sample)

        # Wasserstein GAN loss
        self.dis_loss = tf.reduce_mean(self.dis_real) - tf.reduce_mean(self.dis_fake)
        self.gen_loss = -tf.reduce_mean(self.dis_fake)

        # Optimizer for discriminator
        self.var_dis = [self.weights[i] for i in self.weights if re.match('d', i)] + [self.biases[i] for i in
                                                                                      self.biases if
                                                                                      re.match('d', i)]
        self.clip_dis = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.var_dis]
        self.dis_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.3).minimize(-self.dis_loss,
                                                                                            var_list=self.var_dis)
        # Optimizer for generator parameters
        self.var_gen = [self.weights[i] for i in self.weights if re.match('g', i)] + [self.biases[i] for i in
                                                                                      self.biases if
                                                                                      re.match('g', i)]
        self.gen_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.3).minimize(self.gen_loss,
                                                                                            var_list=self.var_gen)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Create session and graph, initial variables
        self.sess.run(init)

        self.weightSaver = tf.train.Saver(var_list=self.weights)
        self.biaseSaver = tf.train.Saver(var_list=self.biases)

        if os.path.exists(self.path_save + '/DCGAN_weights.ckpt.index'):
            self.weightSaver.restore(self.sess, "./saved_model/DCGAN_weights.ckpt")
            self.biaseSaver.restore(self.sess, "./saved_model/DCGAN_biases.ckpt")

    def load_data(self):
        orig_img = np.empty((0, self.input_height, self.input_width, self.rgb_filters), dtype='float32')

        for pic in glob(self.path_data + '/*.png'):
            img = mpimg.imread(pic)
            # remove alpha channel  %some alpha=0 but RGB is not equal to [1., 1., 1.]
            img[img[:, :, 3] == 0] = np.ones((1, 4))
            img = img[:, :, 0:3]
            orig_img = np.append(orig_img, [img], axis=0)
        return orig_img

    def conv2d(self, x, W, b):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, self.strides, self.strides, 1], padding='SAME')
        return tf.nn.bias_add(x, b)

    def deconv2d(self, x, W, b, out_shape):
        x = tf.nn.conv2d_transpose(x, W, out_shape, strides=[1, self.strides, self.strides, 1],
                                   padding='SAME')
        return tf.nn.bias_add(x, b)

    def lrelu(self, x, leak=0.2, name="lrelu"):
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)

    def project(self, x, W, b):
        return tf.add(tf.matmul(x, W), b)

    def sample_Z(self, m, n):
        return np.random.uniform(-1., 1., size=[m, n])

    def generator(self, z):
        hidden_g1 = self.project(z, self.weights['g_w1'], self.biases['g_w1'])
        hidden_g1 = tf.reshape(hidden_g1, [-1, self.input_height // 8, self.input_width // 8, self.n_filters[2]])

        output_dim2 = tf.stack([tf.shape(z)[0], self.input_height // 4, self.input_width // 4, self.n_filters[1]])
        hidden_g2 = tf.nn.relu(self.deconv2d(hidden_g1, self.weights['g_w2'], self.biases['g_w2'], output_dim2))

        output_dim3 = tf.stack([tf.shape(z)[0], self.input_height // 2, self.input_height // 2, self.n_filters[0]])
        hidden_g3 = tf.nn.relu(self.deconv2d(hidden_g2, self.weights['g_w3'], self.biases['g_w3'], output_dim3))

        output_dim4 = tf.stack([tf.shape(z)[0], self.input_height, self.input_height, self.rgb_filters])
        hidden_g4 = tf.nn.tanh(self.deconv2d(hidden_g3, self.weights['g_w4'], self.biases['g_w4'], output_dim4))
        return hidden_g4

    def discriminator(self, x):
        hidden_d1 = self.lrelu(self.conv2d(x, self.weights['d_w1'], self.biases['d_w1']))
        hidden_d2 = self.lrelu(self.conv2d(hidden_d1, self.weights['d_w2'], self.biases['d_w2']))
        hidden_d3 = self.lrelu(self.conv2d(hidden_d2, self.weights['d_w3'], self.biases['d_w3']))
        hidden_d3 = tf.reshape(hidden_d3, [-1, self.size_filter * self.size_filter * self.n_filters[2]])

        hidden_d4 = self.project(hidden_d3, self.weights['d_w4'], self.biases['d_w4'])
        return tf.nn.sigmoid(hidden_d3), hidden_d3

    def train(self):

        total_batch = int(self.images.shape[0] / self.batch_size)
        # Training cycle
        for epoch in range(self.num_epochs):
            # Loop over all batches
            start = 0
            end = self.batch_size
            for i in range(total_batch - 1):
                index = np.arange(start, end)
                np.random.shuffle(index)
                batch_xs = self.images[index]
                batch_zs = self.sample_Z(self.batch_size, self.gen_dim)
                # Run optimization op and loss op (to get loss value)
                _, self.d_loss_train, _ = self.sess.run([self.dis_optimizer, self.dis_loss, self.clip_dis],
                                                   feed_dict={self.X: batch_xs, self.Z: batch_zs})
                _, self.g_loss_train = self.sess.run([self.gen_optimizer, self.gen_loss], feed_dict={self.Z: batch_zs})
                start = end
                end = start + self.batch_size
            # Display logs per epoch step
            try:
                if ((epoch == 1) or (epoch + 1) % self.display_step == 0) or ((epoch + 1) == self.num_epochs):
                    print('Epoch: {0:05d}      Discriminator loss: {1:f}      Generator loss: {2:f}'.format(epoch + 1,
                                                                                                            self.d_loss_train,
                                                                                                            self.g_loss_train))
            except:
                pass
            if ((epoch == 1) or (epoch + 1) % self.save_step == 0):
                save_path = self.weightSaver.save(self.sess, self.path_save + "/DCGAN_weights.ckpt")
                save_path = self.biaseSaver.save(self.sess, self.path_save + "/DCGAN_biases.ckpt")
        print("Optimization Finished!")

    def generate(self, num):
        Z = tf.placeholder(tf.float32, [None, self.gen_dim])
        # Construct discriminator and generator
        gen_sample = self.generator(Z)
        return self.sess.run(gen_sample, feed_dict={Z: self.sample_Z(num, self.gen_dim)})
