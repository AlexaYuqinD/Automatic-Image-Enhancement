# coding: utf-8

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

import load_vgg
import utils


def setup():
    """
    :checkpoints: model checkpoints 
    :outputs: results
    """
    utils.safe_mkdir("checkpoints")
    utils.safe_mkdir("outputs")


class StyleTransfer(object):
    def __init__(self, content_photo, style_photo, photo_width, photo_height):
        """
        Initialization
        
        :param content_photo: photos to transfer
        :param style_photo: photos with styles
        :param photo_width: width of photos
        :param photo_height: height of photos
        """
        # get information of photos
        self.content_name = str(content_photo.split("/")[-1].split(".")[0])
        self.style_name = str(style_photo.split("/")[-1].split(".")[0])
        self.photo_width = photo_width
        self.photo_height = photo_height
        # set width and height
        self.content_photo = utils.get_resized_photo(content_photo, photo_width, photo_height)
        self.style_photo = utils.get_resized_photo(style_photo, photo_width, photo_height)
        # does not include noise
        self.initial_photo = self.content_photo
        #self.initial_photo = utils.generate_noise_photo(self.content_photo, photo_width, photo_height)

        # define layers
        self.content_layer = "conv4_2"
        self.style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

        # set weights of losses
        self.content_w = 0.001
        self.style_w = 1

        # set weights of different style layers
        self.style_layer_w = [0.5, 1.0, 1.5, 3.0, 4.0]

        # set global step and learning rate
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")  # global step
        self.lr = 2.0

        utils.safe_mkdir("outputs/%s_%s" % (self.content_name, self.style_name))

    def create_input(self):
        """
        Initialize photo tensor
        """
        with tf.variable_scope("input"):
            self.input_photo = tf.get_variable("in_photo", 
                                             shape=([1, self.photo_height, self.photo_width, 3]),
                                             dtype=tf.float32,
                                             initializer=tf.zeros_initializer())

    def load_vgg(self):
        """
        load vgg model to preprocess photos
        """
        self.vgg = load_vgg.VGG(self.input_photo)
        self.vgg.load()
        # mean-center
        self.content_photo -= self.vgg.mean_pixels
        self.style_photo -= self.vgg.mean_pixels

    def _content_loss(self, P, F):
        """
        compute content loss
        
        :param P: feature map of content photo
        :param F: feature map of generalized photo
        """
        self.content_loss = tf.reduce_sum(tf.square(F - P)) / (4.0 * P.size)
        
    def _gram_matrix(self, F, N, M):
        """
        construct Gram Matrix of F，F is a feature map，shape=(widths, heights, channels)
        
        :param F: feature map
        :param N: 3rd dimension of feature map
        :param M: 1st dimension * 2nd dimension of feature map
        :return: Gram Matrix of F
        """
        F = tf.reshape(F, (M, N))

        return tf.matmul(tf.transpose(F), F)

    def _single_style_loss(self, a, g):
        """
        compute style loss of a single layer
        
        :param a: feature map of style photo of current layer
        :param g: feature map of content photo of current layer
        :return: style loss
        """
        N = a.shape[3]
        M = a.shape[1] * a.shape[2]

        # 生成feature map的Gram Matrix
        A = self._gram_matrix(a, N, M)
        G = self._gram_matrix(g, N, M)

        return tf.reduce_sum(tf.square(G - A)) / ((2 * N * M) ** 2)

    def _style_loss(self, A):
        """
        compute total style loss
        
        :param A: all feature maps of style photos
        """
        # num of layers (conv1_1, conv2_1, conv3_1, conv4_1, conv5_1)
        n_layers = len(A)
        # compute loss
        E = [self._single_style_loss(A[i], getattr(self.vgg, self.style_layers[i]))
             for i in range(n_layers)]
        # weighted sum
        self.style_loss = sum(self.style_layer_w[i] * E[i] for i in range(n_layers))

    def losses(self):
        """
        compute losses of model
        """
        with tf.variable_scope("losses"):
            
            # content loss
            with tf.Session() as sess:
                sess.run(self.input_photo.assign(self.content_photo))
                gen_photo_content = getattr(self.vgg, self.content_layer)
                content_photo_content = sess.run(gen_photo_content)
            self._content_loss(content_photo_content, gen_photo_content)

            # style loss
            with tf.Session() as sess:
                sess.run(self.input_photo.assign(self.style_photo))
                style_layers = sess.run([getattr(self.vgg, layer) for layer in self.style_layers])                              
            self._style_loss(style_layers)

            # weighted sum to get total loss
            self.total_loss = self.content_w * self.content_loss + self.style_w * self.style_loss

    def optimize(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss, global_step=self.gstep)

    def create_summary(self):
        with tf.name_scope("summary"):
            tf.summary.scalar("content_loss", self.content_loss)
            tf.summary.scalar("style_loss", self.style_loss)
            tf.summary.scalar("total_loss", self.total_loss)
            self.summary_op = tf.summary.merge_all()

    def build(self):
        self.create_input()
        self.load_vgg()
        self.losses()
        self.optimize()
        self.create_summary()

    def train(self, epochs=300):
        
        skip_step = 1
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter("graphs/style_transfer", sess.graph)
            
            sess.run(self.input_photo.assign(self.initial_photo))

            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname("checkpoints/%s_%s_style_transfer/checkpoint" %
                                                                 (self.content_name, self.style_name)))
            if ckpt and ckpt.model_checkpoint_path:
                print("You have pre-trained model, if you do not want to use this, please delete the existing one.")
                saver.restore(sess, ckpt.model_checkpoint_path)

            initial_step = self.gstep.eval()

            for epoch in range(initial_step, epochs):
                if epoch >= 5 and epoch < 20:
                    skip_step = 10
                elif epoch >= 20:
                    skip_step = 50
                
                sess.run(self.optimizer)
                if (epoch + 1) % skip_step == 0:
                    gen_photo, total_loss, summary = sess.run([self.input_photo,
                                                               self.total_loss,
                                                               self.summary_op])

                    gen_photo = gen_photo + self.vgg.mean_pixels 
                    writer.add_summary(summary, global_step=epoch)

                    print("Step {}\n   Sum: {:5.1f}".format(epoch + 1, np.sum(gen_photo)))
                    print("   Loss: {:5.1f}".format(total_loss))

                    filename = "outputs/%s_%s/epoch_%d.png" % (self.content_name, self.style_name, epoch)
                    utils.save_photo(filename, gen_photo)

                    # save model
                    if (epoch + 1) % 20 == 0:
                        saver.save(sess,
                                   "checkpoints/%s_%s_style_transfer/style_transfer" %
                                   (self.content_name, self.style_name), epoch)

if __name__ == "__main__":
    setup()
    # specify photos
    content_photo = "../data/full/original/a0001.tif"
    style_photo = "../data/full/style/a0001a.tif"
    # specify sizes
    photo_width = 500
    photo_height = 500
    # style transfer
    style_transfer = StyleTransfer(content_photo, style_photo, photo_width, photo_height)
    style_transfer.build()
    style_transfer.train(2000)