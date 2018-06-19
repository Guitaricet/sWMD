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

from time import time

import click
import numpy as np
import pandas as pd

from gensim.models import FastText

import cfg
import functions as f
from datautils import DataLoader

# TODO: make separate logger from the system one
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')


@click.command()
@click.option('--datapath-train', default=cfg.data.datapath_train)
@click.option('--datapath-val', default=cfg.data.datapath_val)
@click.option('--datapath-test', default=cfg.data.datapath_test)
@click.option('--embeddings-path', default=cfg.data.embeddings_path)
@click.option('--savefolder', default=cfg.data.savefolder)
@click.option('--dataset-frac', default=1.0)
def train(datapath_train, datapath_val, datapath_test, embeddings_path, savefolder, dataset_frac):
    results_df = []
    savepath = os.path.join(
        savefolder,
        'lambda{}_lrw{}_lrA{}_maxiter{}_batch{}_nn{}.csv'.format(
            cfg.sinkhorn.lambda_, int(cfg.train.lr_w), int(cfg.train.lr_A),
            cfg.train.max_iter, cfg.train.batch_size, cfg.train.n_neighbors
        )
    )

    embeddings = FastText.load_fasttext_format(embeddings_path)

    logging.info('Loading datasets...')
    dataloader_train = DataLoader(datapath_train, embeddings, cfg.train.batch_size, frac=dataset_frac)
    dataloader_val = DataLoader(datapath_val, embeddings, cfg.train.batch_size, frac=dataset_frac)
    dataloader_test = DataLoader(datapath_test, embeddings, cfg.train.batch_size, frac=dataset_frac)
    logging.info('Datasets are loaded')

    if cfg.train.use_baseline:
        evaluate_wmd(dataloader_train, dataloader_test, embeddings.vector_size)

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
    A = np.random.rand(cfg.train.hidden_dim, embeddings.vector_size)
    w = np.ones([cfg.data.max_dict_size, 1])

    # TODO: Test learned metric for WCD
    # Dc = f.distance(xtr_center, xte_center)

    logging.info('Starting main optimisation loop')
    for i in range(cfg.train.max_iter):
        logging.info('Iter: {}'.format(i))
        [dw, dA] = f.grad_swmd(dataloader_train,
                               x_train_centers,
                               w,
                               A,
                               cfg.train.batch_size,
                               cfg.train.n_neighbors)
        logging.debug('Gradients are computed')

        # Update w and A
        w -= cfg.train.lr_w * dw
        w[w < cfg.train.w_min] = cfg.train.w_min
        w[w > cfg.train.w_max] = cfg.train.w_max

        A -= cfg.train.lr_A * dA

        if i % cfg.train.save_freq == 0:
            # Compute loss
            logging.info('KNN train')
            loss_train = knn_swmd(dataloader_train, dataloader_train, w, A)
            logging.info('Train knn err: %s' % loss_train)

            logging.info('KNN valid')
            loss_valid = knn_swmd(dataloader_train, dataloader_val, w, A)

            logging.info('Valid knn err: %s' % loss_valid)

            results_df.append({'step': i, 'err_train': np.mean(loss_train), 'err_valid': np.mean(loss_valid)})
            pd.DataFrame(results_df).to_csv(savepath)

        del dw, dA
        gc.collect()

    logging.info('KNN test')
    loss_test = knn_swmd(dataloader_train, dataloader_test, w, A)
    logging.info('Test knn err: %s' % loss_test)


def evaluate_wmd(dataloader_train, dataloader_test, embed_dim):
    """
    Standard (non-supervised) WMD evaluation
    Used as a baseline
    """

    logging.info('baseline WMD evaluation')
    logging.info('KNN test')

    w_baseline = np.ones([cfg.data.max_dict_size, 1])
    A_baseline = np.identity(embed_dim)

    loss_test = knn_swmd(dataloader_train,
                         dataloader_test,
                         w_baseline,
                         A_baseline)

    logging.info('Test error per class: %s' % loss_test)
    logging.info('Test mean error:      %s' % np.mean(loss_test))

    return loss_test


def knn_swmd(dataloader_train, dataloader_test, w, A):
    """
    Computes KNN
    Not time and memory efficient, because computes full distance matrix

    :param A, w: model parameters
    :return: KNN error rate
    """
    start_time = time()

    n_train = len(dataloader_train)
    n_test = len(dataloader_test)

    wmd_dist = np.zeros([n_train, n_test])

    # TODO: fix multiprocessing
    # pool = mul.Pool(processes = 6)
    result = []

    logging.info('Train set size: %s' % n_train)
    for i in range(0, n_train):
        if i % 100 == 0:
            logging.info('KNN iter %s' % i)

        Wi = np.zeros(n_test)

        prep_time = time()
        x_i, bow_i, indices_train_i, _ = dataloader_train[i]
        logging.info('Preprocessing time: %s' % (time() - prep_time))

        bow_i.shape = [np.size(bow_i), 1]
        d_a = bow_i * w[indices_train_i]
        d_a = d_a / sum(d_a)

        inn_cycle_time = time()
        for j in range(0, n_test):
            x_j, bow_j, indices_test_j, _ = dataloader_test[j]
            bow_j.shape = [np.size(bow_j), 1]
            d_b = bow_j * w[indices_test_j][0]
            d_b = d_b / sum(d_b)
            d_b.shape = (np.size(d_b), 1)

            # result.append(pool.apply_async(sinkhorn2, (i, j, A, x_i, x_j, a, b)))

            # WARNING! sinkhorn2 and sinkhorn3 have different sets of return parameters
            metric_time = time()
            result.append(f.sinkhorn2(A, x_i, x_j, d_a, d_b))
            if j == 1:
                logging.info('Metric calculation time: %s' % (time() - metric_time))

        logging.info('Inner cycle total time: %s' % (time() - inn_cycle_time))

    # pool.close()
    # pool.join()

    n = 0
    for res in result:
        # r = res.get()
        r = res
        i = n // n_test
        j = np.mod(n, n_test)
        wmd_dist[i, j] = r[3]
        n += 1

    err = f.knn_fall_back(wmd_dist, dataloader_train.labels, dataloader_test.labels, [5])

    del wmd_dist
    gc.collect()

    return err


if __name__ == '__main__':
    train()
