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
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')

script_start_time = time()

@click.command()
@click.option('--datapath-train', default=cfg.data.datapath_train)
@click.option('--datapath-val', default=cfg.data.datapath_val)
@click.option('--datapath-test', default=cfg.data.datapath_test)
@click.option('--embeddings-path', default=cfg.data.embeddings_path)
@click.option('--savefolder', default=cfg.data.savefolder)
@click.option('--use-baseline', default=cfg.train.use_baseline)
@click.option('--knn-sample-size', default=cfg.train.knn_sample_size)
@click.option('--dataset-frac', default=1.0)
@click.option('--results-prefix', default='')
@click.option('--use-random-hyperparameters', default=False)
def train(datapath_train,
          datapath_val,
          datapath_test,
          embeddings_path,
          savefolder,
          use_baseline,
          knn_sample_size,
          dataset_frac,
          results_prefix,
          use_random_hyperparameters):

    # For hyperparameter search
    if use_random_hyperparameters:
        lr_a = 10 ** random.randint(-3, 2)
        lr_w = 10 ** random.randint(-3, 2)
        knn_sample_size *= random.uniform(0.1, 10)
        logging.info('lr_a = {}, lr_w = {}, knn_sample_size = {}'.format(lr_a, lr_w, knn_sample_size))
    else:
        lr_a = cfg.train.lr_A
        lr_w = cfg.train.lr_w

    results_df = []
    savepath = os.path.join(
        savefolder,
        '{}_{}_lambda{}_lrw{}_lrA{}_maxiter{}_batch{}_nn{}.csv'.format(
            results_prefix, script_start_time,
            cfg.sinkhorn.lambda_, int(lr_w), int(lr_a),
            cfg.train.max_iter, cfg.train.batch_size, cfg.train.n_neighbors
        )
    )

    embeddings = FastText.load_fasttext_format(embeddings_path)

    logging.info('Loading datasets...')
    dataloader_train = DataLoader(datapath_train, embeddings, cfg.train.batch_size, frac=dataset_frac)
    dataloader_val = DataLoader(datapath_val, embeddings, cfg.train.batch_size, frac=dataset_frac)
    dataloader_test = DataLoader(datapath_test, embeddings, cfg.train.batch_size, frac=dataset_frac)
    logging.info('Datasets are loaded')

    if use_baseline:
        logging.info('baseline WMD evaluation')
        error = evaluate_wmd(dataloader_train, dataloader_test, embeddings.vector_size)

        pd.DataFrame([{'baseline_error_rate': error}]).to_csv(
            os.path.join(savefolder, '{}_baseline.csv'.format(results_prefix))
        )

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
        logging.info('SGD iter: {}'.format(i))
        [dw, dA] = f.grad_swmd(dataloader_train,
                               x_train_centers,
                               w,
                               A,
                               cfg.train.batch_size,
                               cfg.train.n_neighbors)
        logging.debug('Gradients are computed')

        # Update w and A
        w -= lr_w * dw
        w[w < cfg.train.w_min] = cfg.train.w_min
        w[w > cfg.train.w_max] = cfg.train.w_max

        A -= lr_a * dA

        if i % cfg.train.save_freq == 0 or i == cfg.train.max_iter:
            # Compute error rate
            logging.info('KNN train')
            error_train = knn_swmd(dataloader_train, dataloader_train, w, A, knn_sample_size)
            logging.info('Train knn err: %s' % error_train)

            logging.info('KNN valid')
            error_valid = knn_swmd(dataloader_train, dataloader_val, w, A, knn_sample_size)

            logging.info('Valid knn err: %s' % error_valid)

            results_df.append({'step': i, 'err_train': error_train, 'err_valid': error_valid, 'knn_sample_size': knn_sample_size})
            pd.DataFrame(results_df).to_csv(savepath)

        del dw, dA
        gc.collect()

    logging.info('KNN test')
    error_test = knn_swmd(dataloader_train, dataloader_test, w, A, knn_sample_size)
    results_df.append({'step': cfg.train.max_iter, 'err_train': [], 'err_valid': error_test, 'knn_sample_size': knn_sample_size})
    logging.info('Test knn err: %s' % error_test)


def evaluate_wmd(dataloader_train, dataloader_test, embed_dim, knn_sample_size=cfg.train.knn_sample_size):
    """
    Standard (non-supervised) WMD evaluation
    Used as a baseline
    """

    logging.info('KNN test')

    w_baseline = np.ones([cfg.data.max_dict_size, 1])
    A_baseline = np.identity(embed_dim)

    loss_test = knn_swmd(dataloader_train,
                         dataloader_test,
                         w_baseline,
                         A_baseline,
                         knn_sample_size)

    logging.info('Test error per knn: %s' % loss_test)
    logging.info('Test mean error:    %s' % np.mean(loss_test))

    return loss_test


def knn_swmd(dataloader_train,
             dataloader_test,
             w,
             A,
             train_sample_size=cfg.train.knn_sample_size,
             ks=None):
    """
    Computes KNN
    Not time and memory efficient, because computes full distance matrix

    Args:
        A, w: model parameters
        ks: list of numbers of neighbors if None cfg.train.ks is used
        train_sample_size: trainset sample size for nearest neighbors (0 < train_sample <= 1)
    Returns:
        KNN error rate
    """
    if ks is None:
        ks = cfg.train.ks

    knn_start_time = time()

    n_train = len(dataloader_train)
    n_test = len(dataloader_test)

    # is inf really OK?
    wmd_dist = np.full([n_test, n_train], np.float32('inf'))

    # TODO: fix multiprocessing
    # pool = mul.Pool(processes = 6)
    result = []

    logging.info('Train set size: %s' % n_train)
    stats_list = []

    for j in range(n_test):
        if j % 100 == 0:
            # logging.info('KNN iter: %s' % j)
            pd.DataFrame(stats_list).to_csv('stats_%s.csv' % int(knn_start_time), index=False)

        prep_time = time()
        x_j, bow_j, indices_test_j, _ = dataloader_test[j]
        logging.debug('Preprocessing time: %s' % (time() - prep_time))
        prep_time = time() - prep_time

        d_b = bow_j.reshape(-1, 1) * w[indices_test_j][0]
        d_b = d_b / sum(d_b)

        train_sample = dataloader_train.sample(train_sample_size)
        trainset_cycle_time = time()
        for i in train_sample:

            x_i, bow_i, indices_train_i, _ = dataloader_train[i]

            d_a = bow_i.reshape(-1, 1) * w[indices_train_i]
            d_a /= sum(d_a)

            metric_time = time()

            # result.append(pool.apply_async(sinkhorn2, (i, j, A, x_i, x_j, a, b)))
            # WARNING! sinkhorn2 and sinkhorn3 have different sets of return parameters

            res = f.sinkhorn2(A, x_i, x_j, d_a, d_b)
            wmd_dist[j, i] = res[3]
            if i == 0:
                logging.debug('Metric calculation time: %s' % (time() - metric_time))
                metric_time = time() - metric_time
        
        logging.debug('Inner cycle total time: %s, iters: %s' % ((time() - trainset_cycle_time), n_test))
        trainset_cycle_time = time() - trainset_cycle_time
        stats_list.append({'prep_time': prep_time, 'trainset_cycle_time': trainset_cycle_time})

    # pool.close()
    # pool.join()
    # for n, res in enumerate(result):
    #     # r = res.get()
    #     r = res
    #     i = n // n_test
    #     j = np.mod(n, n_test)
    #     wmd_dist[i, j] = r[3]

    wmd_dist = wmd_dist.T
    err = f.knn_fall_back(wmd_dist, dataloader_train.labels, dataloader_test.labels, ks)

    del wmd_dist
    gc.collect()

    return err


if __name__ == '__main__':
    train()
