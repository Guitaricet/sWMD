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


@click.command()
@click.option('--datapath-train', default=cfg.data.datapath_train)
@click.option('--datapath-val', default=cfg.data.datapath_val)
@click.option('--datapath-test', default=cfg.data.datapath_test)
@click.option('--embeddings-path', default=cfg.data.embeddings_path)
@click.option('--savefolder', default=cfg.data.savefolder)
@click.option('--test-frac', default=1.0)
def train(datapath_train, datapath_val, datapath_test, embeddings_path, savefolder, test_frac):
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
    dataloader_train = DataLoader(datapath_train, embeddings, cfg.train.batch_size)
    dataloader_val = DataLoader(datapath_val, embeddings, cfg.train.batch_size)
    dataloader_test = DataLoader(datapath_test, embeddings, cfg.train.batch_size, frac=test_frac)
    logging.info('Datasets are loaded')

    if cfg.train.use_baseline:
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
            loss_train = f.knn_swmd(dataloader_train, dataloader_train, w, A)
            logging.info('Train knn err: %s' % loss_train)

            logging.info('KNN valid')
            loss_valid = f.knn_swmd(dataloader_train, dataloader_val, w, A)

            logging.info('Valid knn err: %s' % loss_valid)

            results_df.append({'step': i, 'err_train': np.mean(loss_train), 'err_valid': np.mean(loss_valid)})
            pd.DataFrame(results_df).to_csv(savepath)

        del dw, dA
        gc.collect()

    logging.info('KNN test')
    loss_test = f.knn_swmd(dataloader_train, dataloader_test, w, A)
    logging.info('Test knn err: %s' % loss_test)


if __name__ == '__main__':
    train()
