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
import pandas as pd
import scipy.io as sio

from gensim.models import FastText

import functions as f
from datautils import DataLoader

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')

# TODO: add @check

RAND_SEED = 1

save_path = 'results/'

dataset = 'bbcsport'
MAX_DICT_SIZE = 50000

# Optimization parameters
MAX_ITER = 100  # number of iterations
SAVE_FREQ = 10  # frequency of results saving
BATCH_SIZE = 32  # batch size in batch gradient descent (B in the paper)
N_NEIGHBORS = 200  # neighborhood size (N in the paper)
LR_W = 1e+1  # learning rate for w
LR_A = 1e+0  # learning rate for A
LAMBDA_ = 10  # regularisation parameter (lambda in the paper)
CV_FOLDS = 2
W_MIN = 0.01
W_MAX = 10
BASELINE = False
USE_MATLAB_INPUT = False

# Data parameters
DATAPATH_TRAIN = '/Users/vlyalin/Downloads/SBER_FAQ/sber_faq_train.csv'
DATAPATH_VAL = '/Users/vlyalin/Downloads/SBER_FAQ/sber_faq_val.csv'
DATAPATH_TEST = '/Users/vlyalin/Downloads/SBER_FAQ/sber_faq_test.csv'

EMBEDDINGS_PATH = '/Users/vlyalin/Downloads/lenta_lower_100.bin'


def evaluate_wmd(dataloader_train, dataloader_test, embed_dim):
    """
    Standard (non-supervised) WMD as a baseline
    """

    logging.info('baseline WMD evaluation')
    logging.info('KNN test')

    w_baseline = np.ones([MAX_DICT_SIZE, 1])
    A_baseline = np.identity(embed_dim)

    loss_test = f.knn_swmd(dataloader_train,
                           dataloader_test,
                           w_baseline,
                           LAMBDA_,
                           A_baseline)

    logging.info('Test error per class: %s' % loss_test)
    logging.info('Test mean error:      %s' % np.mean(loss_test))

    return loss_test


if __name__ == '__main__':

    results_cv = np.zeros(CV_FOLDS)
    results_df = []

    for split in range(1, CV_FOLDS + 1):
        save_counter = 0

        Err_v = []
        Err_t = []
        w_all = []
        A_all = []
        if USE_MATLAB_INPUT:
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

            raise ValueError('Do not use MatLab!')
        
        embeddings = FastText.load_fasttext_format(EMBEDDINGS_PATH)

        logging.info('Loading datasets')
        dataloader_train = DataLoader(DATAPATH_TRAIN, embeddings, BATCH_SIZE)
        dataloader_val = DataLoader(DATAPATH_VAL, embeddings, BATCH_SIZE)
        dataloader_test = DataLoader(DATAPATH_TEST, embeddings, BATCH_SIZE)
        logging.info('Datasets are loaded')

        if BASELINE:
            # 300 for embed dim, embeddings loaded from .mat file
            evaluate_wmd(dataloader_train, dataloader_test, 300)

        raise NotImplementedError()
        # TODO: compute document centers 
        # Compute document center (mean word vector of each document)
        x_train_center = np.zeros([embeddings.vector_size, len(y_train)], dtype=np.float)
        for i in range(0, len(y_train)):
            centers = np.dot(x_train[i], BOW_x_train[i].T )/ sum(sum(BOW_x_train[i]))
            centers.shape = centers.size
            x_train_center[:, i] = centers

        x_valid_center = np.zeros([embeddings.vector_size, len(y_val)], dtype=np.float)
        for i in range(0, len(y_val)):
            centers = np.dot(x_val[i], BOW_x_val[i].T)/ sum(sum(BOW_x_val[i]))
            centers.shape = centers.size
            x_valid_center[:, i] = centers

        x_test_center = np.zeros([embeddings.vector_size, len(y_test)], dtype=np.float)
        for i in range(0, len(y_test)):
            ec = np.dot(x_test[i], BOW_x_test[i].T) / sum(sum(BOW_x_test[i]))
            ec.shape = ec.size
            x_test_center[:, i] = ec

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

            if i % SAVE_FREQ == 0:
                # Compute loss
                filename = save_path + dataset + '_' + str(LAMBDA_) + '_' + str(int(LR_W)) \
                           + '_' + str(int(LR_A)) + '_' + str(MAX_ITER) + '_' + str(BATCH_SIZE) \
                           + '_' + str(N_NEIGHBORS) + '_' + str(split)

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

                sio.savemat(filename + '.mat', {'err_v': loss_valid, 'err_t': loss_train, 'w': w, 'A': A})

                results_df.append({'cv_split': split, 'step': i, 'err_train': np.mean(loss_train), 'err_valid': np.mean(loss_valid)})
                pd.DataFrame(results_df).to_csv(filename + '.csv')

            del dw, dA
            gc.collect()

        logging.info('KNN test')
        loss_test = f.knn_swmd(
            x_train, y_train, x_test, y_test, BOW_x_train, BOW_x_test, indices_train, indices_test, w, LAMBDA_, A
        )
        logging.info('Test knn err: %s' % loss_test)

        err_t_cv = loss_train[loss_valid == np.min(loss_valid)]
        results_cv[split-1] = err_t_cv[0]
        sio.savemat(save_path + dataset + '_results', {'results_cv': results_cv})

    logging.error('Cross-validation error: %s' % np.mean(results_cv))
