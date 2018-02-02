import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
import scipy.io as sio
from scipy.sparse.linalg import svds
# from skcuda.linalg import svd as svd_cuda
# import pycuda.gpuarray as gpuarray
# from pycuda.tools import DeviceMemoryPool
from sklearn import cluster
from sklearn.preprocessing import normalize
from munkres import Munkres
import os
import time
import argparse
from functools import reduce
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('--lambda1', type=float, default=1.0)
# according to the paper, for yale lambda2=6.3096, for coil 20 and 100, lambda2=30.0, for orl, lambda2=0.2
parser.add_argument('--lambda2', type=float, default=0.20)  # sparsity cost on C
parser.add_argument('--lambda3', type=float, default=1.0)  # lambda on gan loss
parser.add_argument('--lambda4', type=float, default=0.000001)# lambda on AE L2 regularization


parser.add_argument('--lr',  type=float, default=1e-3)  # learning rate

parser.add_argument('--pretrain', type=int, default=0)  # number of iterations of pretraining
parser.add_argument('--epochs', type=int, default=1000)  # number of epochs to train on eqn3 and eqn3plus
parser.add_argument('--enable-at', type=int, default=300)  # epoch at which to enable eqn3plus
parser.add_argument('--dataset', type=str, default='orl', choices=['yaleb', 'orl', 'coil20', 'coil100'])
parser.add_argument('--interval', type=int, default=10)
parser.add_argument('--save', action='store_true')  # save pretrained model

parser.add_argument('--usebn',          action='store_true')

#parser.add_argument('--alpha',          type=float, default=0.1)

parser.add_argument('--matfile',        default=None)
parser.add_argument('--imgmult',        type=float,     default=1.0)
parser.add_argument('--palpha',         type=float,     default=None)
parser.add_argument('--kernel-size',    type=int,       nargs='+',  default=None)



"""
Example launch commands:
CUDA_VISIBLE_DEVICES=0 python dsc_gan.py yaleb_run1 --pretrain 60000 --epochs 4000 --enable-at 3000 --dataset yaleb
    pretrain for 60000 iterations first, then train on eqn3 for 3000 epochs, and on eqn3plus for 1000 epochs
CUDA_VISIBLE_DEVICES=0 python dsc_gan.py orl_run1   --pretrain 10000 --epochs 4000 --enable-at 2000 --dataset orl
    pretrain for 10000 iterations first, then train on eqn3 for 2000 epochs, and on eqn3plus for 2000 epochs
"""


class ConvAE(object):
    def __init__(self,
                 args,
                 n_input, n_hidden, kernel_size, n_class, n_sample_perclass, 
                 lambda1, lambda2, lambda3, batch_size,
                 reg=None, disc_bound=0.02,
                 model_path=None, restore_path=None,
                 logs_path='logs'):
        self.args = args
        self.n_class = n_class
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.kernel_size = kernel_size
        self.n_sample_perclass = n_sample_perclass
        self.batch_size = batch_size
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0

        """
        Shared

        x           spatial input
        mask        spatial mask
        latent      spatial code
        """
        # input required to be fed
        self.x    = tf.placeholder(tf.float32, [None, n_input[0], n_input[1], 1])
        self.mask = tf.placeholder(tf.float32, self.x.shape)
        self.learning_rate = tf.placeholder(tf.float32, [])

        # run input through encoder, latent is the output, shape is the shape of encoder
        latent, shape = self.encoder(self.x)
        self.latent_shape = latent.shape
        self.latent_size = reduce(lambda x, y: int(x) * int(y), self.latent_shape[1:], 1)
        print 'latent size={}'.format(self.latent_size)

        """
        Pretraining
        """
        # create decoder with latent
        self.x_r_pre = self.decoder(latent, shape, reuse=False)
        diff_pre = tf.subtract(self.x_r_pre, self.x)
        self.loss_recon_pre = 0.5 * tf.reduce_sum(tf.pow(diff_pre * self.mask, 2.0))

        ae_weights = [v for v in tf.trainable_variables() if (v.name.startswith('enc') or v.name.startswith('dec'))]

        # weight decay
        self.loss_aereg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss_pretrain = self.loss_recon_pre + self.loss_aereg
        with tf.variable_scope('optimizer_pre'):
            self.optimizer_pre = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_pretrain, var_list=ae_weights)

        """
        Eqn 3

        Coef        affinity matrix
        z           flat version of 'latent'
        z_c         flat code multiplied by Coef
        latent_c    spatial version of z_c
        """
        # self-expressive layer
        z = tf.reshape(latent, [batch_size, -1])
        z.set_shape([batch_size, self.latent_size])

        if args.usebn:
            z = tf.contrib.layers.batch_norm(z)

        Coef = tf.Variable(1.0e-4 * tf.ones([self.batch_size, self.batch_size], tf.float32), name='Coef')
        z_c = tf.matmul(Coef, z)
        self.Coef = Coef
        Coef_weights = [Coef]
        latent_c = tf.reshape(z_c, tf.shape(latent))  # petential problem here
        self.z = z

        # create decoder with latent_c
        self.x_r = self.decoder(latent_c, shape, reuse=True)
        eqn3_weights = Coef_weights + ae_weights

        """
        Eqn3
        """
        diff = tf.subtract(self.x_r, self.x)
        self.loss_recon = 0.5 * tf.reduce_sum(tf.pow(diff * self.mask, 2.0))
        #self.loss_sparsity = tf.reduce_sum(tf.pow(self.Coef, 2.0))
        self.loss_sparsity = tf.reduce_sum(tf.abs(self.Coef))
        self.loss_selfexpress = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(z_c, z), 2.0))
        self.loss_eqn3 = self.loss_recon + lambda1 * self.loss_sparsity + lambda2 * self.loss_selfexpress + self.loss_aereg
        with tf.variable_scope('optimizer_eqn3'):
            self.optimizer_eqn3 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_eqn3, var_list=eqn3_weights)

        """
        Eqn3+
        """
        # Eqn 3 + K-means
        kmeans_weight, sumdiff, membership = self.kmeans(z, init_centroids)
        self.kmeans_assign 
        self.loss_kmeans = tf.reduce_sum(tf.multiply(sumdiff, membership))
        self.loss_eqn3plus = self.loss_eqn3 + lambda3 * self.loss_kmeans
        eqn3plus_weights = eqn3_weights + [kmeans_weight]
        with tf.variable_scope('optimizer_eqn3plus'):
            self.optimizer_eqn3plus = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_eqn3plus, var_list=eqn3plus_weights)

        # finalize stuffs
        self.ae_weight_norm = tf.sqrt(sum([tf.norm(v, 2) ** 2 for v in ae_weights]))
        s0 = tf.summary.scalar("loss_recon_pre",    self.loss_recon_pre / batch_size)  # 13372
        s1 = tf.summary.scalar("loss_recon",        self.loss_recon)
        s2 = tf.summary.scalar("loss_sparsity",     self.loss_sparsity)
        s3 = tf.summary.scalar("loss_selfexpress",  self.loss_selfexpress)
        s5 = tf.summary.scalar("ae_l2_norm",        self.ae_weight_norm)  # 29.8
        s6 = tf.summary.scalar("loss_kmeans",       self.loss_kmeans)
        self.summaryop_pretrain = tf.summary.merge([s0, s5])
        self.summaryop_eqn3     = tf.summary.merge([s1, s2, s3, s5])
        self.summaryop_eqn3plus = tf.summary.merge([s1, s2, s3, s4, s5, s6, s7])
        self.init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # stop TF from eating up all GPU RAM
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess = tf.InteractiveSession(config=config)
        self.sess.run(self.init)
        self.saver = tf.train.Saver([v for v in ae_weights if v.name.startswith('enc_w') or v.name.startswith('dec_w')])
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph(), flush_secs=20)

    # Building the encoder
    def encoder(self, x):
        shapes = []
        n_hidden = [1] + self.n_hidden
        input = x
        for i, k_size in enumerate(self.kernel_size):
            w = tf.get_variable('enc_w{}'.format(i), shape=[k_size, k_size, n_hidden[i], n_hidden[i + 1]],
                                initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
            b = tf.get_variable('enc_b{}'.format(i), shape=[n_hidden[i + 1]], initializer=tf.zeros_initializer())
            shapes.append(input.get_shape().as_list())
            enc_i = tf.nn.conv2d(input, w, strides=[1, 2, 2, 1], padding='SAME')
            enc_i = tf.nn.bias_add(enc_i, b)
            enc_i = tf.nn.relu(enc_i)
            input = enc_i
        return input, shapes

    # Building the decoder
    def decoder(self, z, shapes, reuse=False):
        # Encoder Hidden layer with sigmoid activation #1
        input = z
        n_hidden = list(reversed([1] + self.n_hidden))
        shapes = list(reversed(shapes))
        for i, k_size in enumerate(reversed(kernel_size)):
            with tf.variable_scope('', reuse=reuse):
                w = tf.get_variable('dec_w{}'.format(i), shape=[k_size, k_size, n_hidden[i + 1], n_hidden[i]],
                                    initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
                b = tf.get_variable('dec_b{}'.format(i), shape=[n_hidden[i + 1]], initializer=tf.zeros_initializer())
                dec_i = tf.nn.conv2d_transpose(input, w, tf.stack(
                    [tf.shape(self.x)[0], shapes[i][1], shapes[i][2], shapes[i][3]]),
                                               strides=[1, 2, 2, 1], padding='SAME')
                dec_i = tf.add(dec_i, b)
                if i != len(self.n_hidden) - 1:
                    dec_i = tf.nn.relu(dec_i)
                input = dec_i
        return input

    def kmeans(self, z, init_centroids):
        assert init_centroids.shape == (self.n_class, self.latent_size)
        with tf.variable_scope('kmeans'):
            kmeans_weights = tf.get_variable('centroids', shape=[self.n_class, self.latent_size])
        self.kmeans_weights = kmeans_weights
        ez = tf.expand_dims(z, 1)               # N, 1, D
        ek = tf.expand_dims(kmeans_weights, 0)  # 1, K, D
        diff = tf.pow(ez - ek, 2)
        sumdiff = tf.reduce_sum(diff, 2)        # N, K
        membership = tf.nn.softmax(sumdiff)
        return [kmeans_weights], sumdiff, membership

    def partial_fit_eqn3(self, X, lr):
        # take a step on Eqn 3/4
        cost, Coef, summary, _ = self.sess.run((self.loss_recon, self.Coef, self.summaryop_eqn3, self.optimizer_eqn3),
                                               feed_dict={self.x: X, self.learning_rate: lr})
        self.summary_writer.add_summary(summary, self.iter)
        self.iter += 1
        return cost, Coef

    def partial_fit_eqn3plus(self, X, y_x, lr):
        # assert y_x.min() == 0, 'y_x is 0-based'
        cost, Coef, summary, _, _ = self.sess.run(
            [self.loss_recon, self.Coef, self.summaryop_eqn3plus, self.optimizer_eqn3plus, self.gen_step_op],
            feed_dict={self.x: X, self.y_x: y_x, self.learning_rate: lr})
        self.summary_writer.add_summary(summary, self.iter)
        self.iter += 1
        return cost, Coef

    def partial_fit_pretrain(self, X, lr):
        cost, summary, _ = self.sess.run([self.loss_recon_pre, self.summaryop_pretrain, self.optimizer_pre],
                                         feed_dict={self.x: X, self.learning_rate: lr})
        self.summary_writer.add_summary(summary, self.iter)
        self.iter += 1
        return cost

    def get_ae_weight_norm(self):
        norm, = self.sess.run([self.ae_weight_norm])
        return norm

    def get_loss_recon_pre(self, X):
        loss_recon_pre, = self.sess.run([self.loss_recon_pre], feed_dict={self.x: X})
        return loss_recon_pre

    def log_accuracy(self, accuracy):
        summary = tf.Summary(value=[tf.Summary.Value(tag='accuracy', simple_value=accuracy)])
        self.summary_writer.add_summary(summary, self.iter)

    def initlization(self):
        self.sess.run(self.init)

    def reconstruct(self, X):
        return self.sess.run(self.x_r, feed_dict={self.x: X})

    def transform(self, X):
        return self.sess.run(self.z, feed_dict={self.x: X})

    def save_model(self):
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print("model restored")

    def check_size(self, X):
        z = self.sess.run(self.z, feed_dict={self.x: X})
        return z


def best_map(L1, L2):
    # L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def build_aff(C):
    N = C.shape[0]
    Cabs = np.abs(C)
    ind = np.argsort(-Cabs, 0)
    for i in range(N):
        Cabs[:, i] = Cabs[:, i] / (Cabs[ind[0, i], i] + 1e-6)
    Cksym = Cabs + Cabs.T;
    return Cksym


def spectral_cluster(L, n, eps=2.2 * 10 - 8):
    """
    L: Laplacian
    n: number of clusters
    Translates MATLAB code below:
    N  = size(L, 1)
    DN = diag( 1./sqrt(sum(L)+eps) );
    LapN = speye(N) - DN * L * DN;
    [~,~,vN] = svd(LapN);
    kerN = vN(:,N-n+1:N);
    normN = sum(kerN .^2, 2) .^.5;
    kerNS = bsxfun(@rdivide, kerN, normN + eps);
    groups = kmeans(kerNS,n,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
    """
    N = L.shape[0]
    DN = (1. / np.sqrt(L.sum(0) + eps))
    LapN = np.eye(N) - DN * L * DN


def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = d * K + 1  # K=38, d=10
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    # U, S, _ = svd_cuda(C, allocator=mem_pool)
    # take U and S from GPU
    # U = U[:, :r].get()
    # S = S[:r].get()
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    #Z = Z * (Z > 0)
    L = np.abs(np.abs(Z) ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L)  # +1
    return grp, L


def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate


def build_laplacian(C):
    C = 0.5 * (np.abs(C) + np.abs(C.T))
    W = np.sum(C, axis=0)
    W = np.diag(1.0 / W)
    L = W.dot(C)
    return L


def reinit_and_optimize(args, Img, Label, CAE, n_class, k=10, post_alpha=3.5, alpha = 0.1):

    best_epoch=0
    best_acc=0
    best_alpha=0
    best_postalpha=0
    #alpha = args.alpha #max(0.4 - (n_class - 1) / 10 * 0.1, 0.1)
    print(alpha)

    acc_ = []

    if args.epochs is None:
        num_epochs = 50 + n_class * 25  # 100+n_class*20
    else:
        num_epochs = args.epochs

    # init
    CAE.initlization()
    bn=15

    ###
    ### Stage 1: pretrain
    ###
    # if we skip pretraining, we restore already-trained model
    if args.pretrain == 0:
        CAE.restore()
        Z = CAE.sess.run(CAE.z, feed_dict={CAE.x: Img})
        sio.savemat('orl_Z.mat', dict(Z=Z))
    # otherwise we pretrain the model first
    else:
        print('Pretrain for {} steps'.format(args.pretrain))
        """
        After pretrain: 
            AE l2 norm   : 29
            Ae recon loss: 13372
        """
        for epoch in range(1, args.pretrain + 1):
            minibatch_size = 128
            indices = np.random.permutation(Img.shape[0])[:minibatch_size]
            minibatch = Img[indices]  # pretrain with random mini-batch
            cost = CAE.partial_fit_pretrain(minibatch, args.lr) # FIXME provide mask
            if epoch % 100 == 0:
                norm = CAE.get_ae_weight_norm()
                print('pretraining epoch {}, cost: {}, norm: {}'.format(epoch, cost / float(minibatch_size), norm))
        if args.save:
            CAE.save_model()
    ###
    ### Stage 2: fine-tune network
    ###
    print('Finetune for {} steps'.format(num_epochs))
    acc_x = 0.0
    y_x_mode = 'svd'
    for epoch in range(1, num_epochs + 1):
        # eqn3
        if epoch < args.enable_at:
            cost, Coef = CAE.partial_fit_eqn3(Img, args.lr) # FIXME provide mask
        # every interval epochs, perform clustering and evaluate accuracy
        if epoch % args.interval == 0:
            print("epoch: %.1d" % epoch, "cost: %.8f" % (cost / float(batch_size)))
            Coef = thrC(Coef, alpha)
            t_begin = time.time()
            print('================================================')
            print('Warning: clustering produced empty clusters')
            print('================================================')
            missrate_x = err_rate(Label, y_x)
            t_end = time.time()
            acc_x = 1 - missrate_x
            print("accuracy: {}".format(acc_x))
            print('post processing time: {}'.format(t_end - t_begin))
            CAE.log_accuracy(acc_x)
            clustered = True
            #if epoch < 300 and acc_x> 0.85 and acc_x<0.865:
            #   sio.savemat('orl_label_nisp4.mat', dict(s=y_x_new))
            if best_acc < acc_x:
               best_acc = acc_x
               sio.savemat('orl_label_nips_l1.mat', dict(s=y_x_new))

    mean = acc_x
    median = acc_x
    print("{} subjects, accuracy: {}".format(n_class, acc_x))

    return (1 - mean), (1 - median), best_epoch, best_acc, best_alpha, best_postalpha


def prepare_data_YaleB(folder):
    # load face images and labels
    mat = sio.loadmat(os.path.join(folder, 'Yale.mat'))
    img = mat['Y']

    # Reorganize data a bit, put images into Img, and labels into Label
    I = []
    Label = []
    for i in range(img.shape[2]):       # i-th subject
        for j in range(img.shape[1]):   # j-th picture of i-th subject
            temp = np.reshape(img[:,j,i],[42,48])
            Label.append(i)
            I.append(temp)
    I = np.array(I)
    Label = np.array(Label[:])
    Img = np.transpose(I,[0,2,1])
    Img = np.expand_dims(Img[:],3)

    # constants
    n_input = [48,42]
    n_hidden = [10,20,30]
    kernel_size = [5,3,3]
    n_sample_perclass = 64
    disc_size = [200,50,1]
    # tunable numbers
    k=10
    post_alpha=3.5
    alpha = 0.1

    all_subjects = [38] # number of subjects to use in experiment
    model_path   = os.path.join(folder, 'yale-model.ckpt')
    return alpha, Img, Label, n_input, n_hidden, kernel_size, n_sample_perclass, disc_size, k, post_alpha, all_subjects, model_path


def prepare_data_orl(folder):
    mat = sio.loadmat(os.path.join(folder, 'ORL.mat'))
    Label = mat['gnd'].reshape(400).astype(np.int32)
    Img = mat['fea'].reshape(400, 32, 32, 1)

    # constants
    n_input  = [32, 32]
    n_hidden = [3, 3, 5]
    kernel_size = [3, 3, 3]
    n_sample_perclass = 10
    disc_size = [200,50,1]
    # tunable numbers
    k= 3             # svds parameter
    post_alpha= 2.0  # Laplacian parameter
    alpha = 0.2

    all_subjects=[40]
    model_path  = os.path.join(folder, 'orl-model.ckpt')
    return alpha, Img, Label, n_input, n_hidden, kernel_size, n_sample_perclass, disc_size, k, post_alpha, all_subjects, model_path


def prepare_data_coil20(folder):
    mat = sio.loadmat(os.path.join(folder, 'COIL20.mat'))
    Label = mat['gnd'].reshape(-1).astype(np.int32) # 1440
    Img = mat['fea'].reshape(-1, 32, 32, 1)
    #Img = normalize_data(Img)

    # constants
    n_input  = [32, 32]
    n_hidden = [15]
    kernel_size = [3]
    n_sample_perclass = Img.shape[0] / 20
    disc_size = [50,1]
    # tunable numbers
    k=12            # svds parameter
    post_alpha=8.0  # Laplacian parameter
    alpha = 0.04

    all_subjects=[20]
    model_path  = os.path.join(folder, 'coil20-model15.ckpt')
    return alpha, Img, Label, n_input, n_hidden, kernel_size, n_sample_perclass, disc_size, k, post_alpha, all_subjects, model_path


def prepare_data_coil100(folder):
    mat = sio.loadmat(os.path.join(folder, 'COLT100.mat'))
    Label = mat['gnd'].reshape(-1).astype(np.int32) # 1440
    Img = mat['fea'].reshape(-1, 32, 32, 1)

    # constants
    n_input  = [32, 32]
    n_hidden = [50]
    kernel_size = [5]
    n_sample_perclass = Img.shape[0] / 100
    disc_size = [50,1]
    # tunable numbers
    k=12            # svds parameter
    post_alpha=8.0  # Laplacian parameter
    alpha = 0.04

    all_subjects=[100]
    model_path  = os.path.join(folder, 'coil100-model50.ckpt')
    return alpha, Img, Label, n_input, n_hidden, kernel_size, n_sample_perclass, disc_size, k, post_alpha, all_subjects, model_path


def normalize_data(data):
    data = data - data.mean(axis=0)
    data = data / data.std(axis=0)
    return data


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.name is not None and args.name != '', 'name of experiment must be specified'

    # prepare data
    folder = os.path.dirname(os.path.abspath(__file__))
    preparation_funcs = {
        'yaleb': prepare_data_YaleB,
        'orl': prepare_data_orl,
        'coil20': prepare_data_coil20,
        'coil100': prepare_data_coil100}
    assert args.dataset in preparation_funcs
    alpha, Img, Label, n_input, n_hidden, kernel_size, n_sample_perclass, disc_size, k, post_alpha, all_subjects, model_path = \
            preparation_funcs[args.dataset](folder)
    Img = Img*args.imgmult
    post_alpha = args.palpha or post_alpha
    logs_path = os.path.join(folder, 'logs', args.name)
    restore_path = model_path

    # arrays for logging results
    avg = []
    med = []

    # for each experiment setting, perform one loop
    for n_class in all_subjects:
        batch_size = n_class * n_sample_perclass

        lambda1 = args.lambda1  # L2 sparsity on C
        lambda2 = args.lambda2  # 0.2 # 1.0 * 10 ** (n_class / 10.0 - 3.0)    # self-expressivity
        lambda3 = args.lambda3  # k-means weight

        # clear graph and build a new conv-AE
        tf.reset_default_graph()
        CAE = ConvAE(
            args,
            n_input, n_hidden, kernel_size, n_class, n_sample_perclass, 
            lambda1, lambda2, lambda3, batch_size, 
            reg=tf.contrib.layers.l2_regularizer(tf.ones(1) * args.lambda4), disc_bound=args.bound,
            model_path=model_path, restore_path=restore_path, logs_path=logs_path)

        # perform optimization
        avg_i, med_i, best_epoch, best_acc, best_alpha, best_postalpha = reinit_and_optimize(args, Img, Label, CAE, n_class, k=k, post_alpha=post_alpha, alpha = alpha)
        # add result to list
        avg.append(avg_i)
        med.append(med_i)

    # report results for all experiments
    for i, n_class in enumerate(all_subjects):
        print('%d subjects:' % n_class)
        print('Mean: %.4f%%' % (avg[i] * 100), 'Median: %.4f%%' % (med[i] * 100))

        print( best_epoch , best_acc, best_alpha, best_postalpha)
