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
MAX_ITER = 100  # number of stochastic gradient steps
SAVE_FREQ = 10  # frequency of results saving
BATCH_SIZE = 32  # batch size in batch gradient descent (B in the paper)
N_NEIGHBORS = 200  # neighborhood size (N in the paper)
LR_W = 1e+1  # learning rate for w
LR_A = 1e+0  # learning rate for A
LAMBDA_ = 10  # regularisation parameter (lambda in the paper)
CV_FOLDS = 2
W_MIN = 0.01
W_MAX = 10
BASELINE = True
USE_MATLAB_INPUT = False
HIDDEN_DIM = 32

# Data parameters
DATAPATH_TRAIN = '/Users/vlyalin/Downloads/SBER_FAQ/sber_faq_train.csv'
DATAPATH_VAL = '/Users/vlyalin/Downloads/SBER_FAQ/sber_faq_val.csv'
DATAPATH_TEST = '/Users/vlyalin/Downloads/SBER_FAQ/sber_faq_test.csv'

EMBEDDINGS_PATH = '/Users/vlyalin/Downloads/lenta_lower_100.bin'


def evaluate_wmd(dataloader_train, dataloader_test, embed_dim):
    """
    Standard (non-supervised) WMD evaluation
    Used as a baseline
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
        
        embeddings = FastText.load_fasttext_format(EMBEDDINGS_PATH)

        logging.info('Loading datasets...')
        dataloader_train = DataLoader(DATAPATH_TRAIN, embeddings, BATCH_SIZE)
        dataloader_val = DataLoader(DATAPATH_VAL, embeddings, BATCH_SIZE)
        dataloader_test = DataLoader(DATAPATH_TEST, embeddings, BATCH_SIZE)
        logging.info('Datasets are loaded')

        if BASELINE:
            f.evaluate_wmd(dataloader_train, dataloader_test, embeddings.vector_size)

        # Compute document center (mean word vector of each document)
        logging.info('Computing document centers...')
        x_train_centers = np.zeros([embeddings.vector_size, len(dataloader_train)], dtype=np.float)
        for i in range(len(dataloader_train)):
            x, bow, _, _ = dataloader_train[i]
            centers = np.dot(x.T, bow) / sum(bow)
            x_train_centers[:, i] = centers
        logging.info('Document centers are computed')

        # Define optimization parameters:
        # TODO: initialize A via WCD (word centroid distance) training
        A = np.random.rand(shape=(HIDDEN_DIM, embeddings.vector_size))
        w = np.ones([MAX_DICT_SIZE, 1])  # why ones initialisation?

        # TODO: Test learned metric for WCD
        # Dc = f.distance(xtr_center, xte_center)

        logging.info('Starting main optimisation loop')
        for i in range(1, MAX_ITER + 1):
            logging.info('Dataset: {}, split: {}, iter: {}'.format(dataset, split, i))
            [dw, dA] = f.grad_swmd(dataloader_train,
                                   w,
                                   A,
                                   LAMBDA_,
                                   BATCH_SIZE,
                                   N_NEIGHBORS)
            logging.debug('Gradients are computed')

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
                loss_train = f.knn_swmd(dataloader_train, dataloader_train, w, LAMBDA_, A)
                logging.info('Train knn err: %s' % loss_train)

                logging.info('KNN valid')
                loss_valid = f.knn_swmd(dataloader_train, dataloader_val, w, LAMBDA_, A)

                logging.info('Valid knn err: %s' % loss_valid)

                sio.savemat(filename + '.mat', {'err_v': loss_valid, 'err_t': loss_train, 'w': w, 'A': A})

                results_df.append({'cv_split': split, 'step': i, 'err_train': np.mean(loss_train), 'err_valid': np.mean(loss_valid)})
                pd.DataFrame(results_df).to_csv(filename + '.csv')

            del dw, dA
            gc.collect()

        logging.info('KNN test')
        loss_test = f.knn_swmd(
            dataloader_train, dataloader_test, w, LAMBDA_, A
        )
        logging.info('Test knn err: %s' % loss_test)

        err_t_cv = loss_train[loss_valid == np.min(loss_valid)]
        results_cv[split-1] = err_t_cv[0]
        sio.savemat(save_path + dataset + '_results', {'results_cv': results_cv})

    logging.error('Cross-validation error: %s' % np.mean(results_cv))
