"""
Supervised Word-Mover Distance

Based on Boyuan Pan sWMD, https://github.com/ByronPan/sWMD which itself based on https://github.com/gaohuang/S-WMD


Just in case, old MatLab comment on data format:

X is a cell array of all documents, each represented by a dxm matrix where d is the dimensionality of the word embedding and m is the number of unique words in the document
Y is an array of labels
BOW_X is a cell array of word counts for each document
indices is a cell array of global unique IDs for words in a document
TR is a matrix whose ith row is the ith training split of document indices
TE is a matrix whose ith row is the ith testing split of document indices
"""

import os
import gc
import sys
import random
import logging
import pyximport
#pyximport.install(reload_support=True)

import numpy as np
import scipy.io as sio

import functions as f

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')

RAND_SEED = 1

save_path = 'results/'

dataset = 'bbcsport'
MAX_DICT_SIZE = 50000

# Optimization parameters
MAX_ITER = 100  # number of iterations
SAVE_FREQ = MAX_ITER  # frequency of saving results
BATCH_SIZE = 32  # batch size in batch gradient descent (B in the paper)
N_NEIGHBORS = 200  # neighborhood size (N in the paper)
LR_W = 1e+1  # learning rate for w
LR_A = 1e+0  # learning rate for A
LAMBDA_ = 10  # regularisation parameter (lambda in the paper)
CV_FOLDS = 2
W_MIN = 0.01
W_MAX = 10


if __name__ == '__main__':

    results_cv = np.zeros(CV_FOLDS)

    for split in range(1, CV_FOLDS + 1):
        save_counter = 0

        Err_v = []
        Err_t = []
        w_all = []
        A_all = []
        [x_trainval, x_test, y_trainval, y_test, BOW_x_trainval, BOW_x_test, indices_trainval, indices_test] = f.load_data(dataset, split - 1)
        [idx_tr, idx_val] = f.makesplits(y_trainval, 1 - 1.0 / CV_FOLDS, 1, 0)

        x_val = x_trainval[idx_val]
        y_val = y_trainval[idx_val]
        BOW_x_val = BOW_x_trainval[idx_val]
        indices_val = indices_trainval[idx_val]

        x_train = x_trainval[idx_tr]
        y_train = y_trainval[idx_tr]

        BOW_x_train = BOW_x_trainval[idx_tr]
        indices_train = indices_trainval[idx_tr]

        dim = np.size(x_train[0], 0)  # dimension of word vector

        # Compute document center (mean word vector of each document)
        x_train_center = np.zeros([dim, len(y_train)], dtype=np.float)
        for i in range(0, len(y_train)):
            centers = np.dot(x_train[i], BOW_x_train[i].T )/ sum(sum(BOW_x_train[i]))
            centers.shape = centers.size
            x_train_center[:, i] = centers

        xv_center = np.zeros([dim, len(y_val)], dtype=np.float)
        for i in range(0, len(y_val)):
            centers = np.dot(x_val[i], BOW_x_val[i].T)/ sum(sum(BOW_x_val[i]))
            centers.shape = centers.size
            xv_center[:, i] = centers

        xte_center = np.zeros([dim, len(y_test)], dtype=np.float)
        for i in range(0, len(y_test)):
            ec = np.dot(x_test[i], BOW_x_test[i].T) / sum(sum(BOW_x_test[i]))
            ec.shape = ec.size
            xte_center[:, i] = ec

        # Load initialize A (train with WCD — word centroid distance)
        bbc_ini = sio.loadmat('metric_init/' + dataset + '_seed' + str(split) + '.mat')
        A = bbc_ini['Ascaled']

        # Define optimization parameters
        w = np.ones([MAX_DICT_SIZE, 1])

        # Test learned metric for WCD   TO BE CONTINUED!! (Oh, rly?)
        # Dc = f.distance(xtr_center, xte_center)

        # Main loop
        for i in range(1, MAX_ITER + 1):
            logging.info('Dataset: {}, split: {}, iter: {}'.format(dataset, split, i))
            [dw, dA] = f.grad_swmd(x_train,
                                   y_train,
                                   BOW_x_train,
                                   indices_train,
                                   x_train_center,
                                   w,
                                   A,
                                   LAMBDA_,
                                   BATCH_SIZE,
                                   N_NEIGHBORS)
            logging.debug('Gradients are computed')

            # raw_input(np.size(dw))
            # raw_input(np.size(w))

            # Update w and A
            w -= LR_W * dw
            w[w < W_MIN] = W_MIN
            w[w > W_MAX] = W_MAX

            A -= LR_A * dA

            if i % SAVE_FREQ == 0 or i == 1 or i == 3 or i == 10 or i == 50 or i == 200:
                # Compute loss
                filename = save_path + dataset + '_' + str(LAMBDA_) + '_' + str(int(LR_W)) \
                           + '_' + str(int(LR_A)) + '_' + str(MAX_ITER) + '_' + str(BATCH_SIZE) \
                           + '_' + str(N_NEIGHBORS) + '_' + str(split) + '.mat'

                logging.info('KNN train')
                loss_train = f.knn_swmd(
                    x_train, y_train, x_train, y_train, BOW_x_train, BOW_x_train, indices_train, indices_train, w, LAMBDA_, A
                )
                logging.info('Train knn err: %s' % loss_train)

                logging.info('KNN valid')
                loss_valid = f.knn_swmd(
                    x_train, y_train, x_val, y_val, BOW_x_train, BOW_x_val, indices_train, indices_val, w, LAMBDA_, A
                )

                logging.info('Valid knn err: %s' % loss_valid)
                save_counter += 1
                sio.savemat(filename, {'err_v': loss_valid, 'err_t': loss_train, 'w': w, 'A': A})

            del dw, dA
            gc.collect()

        logging.info('KNN test')
        loss_test = f.knn_swmd(
            x_train, y_train, x_test, y_test, BOW_x_train, BOW_x_val, indices_train, indices_val, w, LAMBDA_, A
        )
        logging.info('Test knn err: %s' % loss_test)

        err_t_cv = loss_train[loss_valid == np.min(loss_valid)]
        results_cv[split-1] = err_t_cv[0]
        sio.savemat(save_path + dataset + '_results', {'results_cv': results_cv})

    logging.error('Cross-validation error: %s' % np.mean(results_cv))
