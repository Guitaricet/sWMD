"""
Functions for data processing, gradient and distances computation

More pythonic version of Boyuan Pan sWMD, https://github.com/ByronPan/sWMD
"""
# TODO: rename all a and b variables

import os
import gc
import random
import logging
import multiprocessing as mul

import numpy as np

import scipy.spatial.distance as sdist

from scipy import stats

import pyximport
pyximport.install(reload_support=True)

cimport numpy as np
cimport cython

import cfg

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')


EPS = 1e-10


def distance(np.ndarray[np.double_t, ndim =2] X, np.ndarray[np.double_t, ndim =2] x):
    # computes the pairwise squared distance matrix
    # between any column vectors in X and in x

    cdef int D, d
    cdef np.ndarray[np.double_t, ndim =2] dist

    D = np.size(X[:,0])
    d = np.size(x[:,0])
    if D != d:
        logging.error('Both sets of vectors must have same dimensionality!')
        os._exit()
    dist = sdist.cdist(X.T, x.T, 'sqeuclidean')

    return dist


def grad_swmd(dataloader, document_centers, w, A, batch_size, n_neighbours):
    """
    Computes gradients with respect to A and w for one random batch
    The formula can be found in https://papers.nips.cc/paper/6139-supervised-word-movers-distance.pdf
    (formula 8 and 10, page 4)

    :param batch: dataloader object with batch format:
        x_train: matrix of documents (embed_dim x doc_idx x n_words)
        bow_x_train: bag-of-words text representations (d in the paper)
        indices_train: indices of words in the text for each text (mostly used for sparce matrix summation)
        y_train: list of classes of documents
    :param document_centers: mean word vectors for each (train) document in A-transformed space
    :param w: words weights
    :param A: transformation matrix
    :param batch_size: batch size
    :param n_neighbours: number of nearest neighbours
    :return: dw, dA - values of gradients, mean gradient for the batch
    """

    dim = dataloader._embeddings.vector_size
    n_train = len(dataloader)  # number of documents

    dw = np.zeros(w.shape)  # (vocab_len, 1)
    dA_aux = np.zeros([dim, dim])  # dA = np.dot(A, dA_aux)

    # Sample documents
    batch_indices = random.sample(range(n_train), batch_size)

    # Euclidean distances between document centers
    D_c = distance(np.dot(A, document_centers), np.dot(A, document_centers))
    tr_loss = 0
    n_nan = 0

    logging.debug('Starting batch iteration')

    for i in batch_indices:
        # i for document index in dataloader

        xi, bow_i, ids_i, yi = dataloader[i]

        ids_i.shape = ids_i.size
        bow_i.shape = (np.size(bow_i), 1)
        d_a_tilde = bow_i * w[ids_i]
        d_a_tilde = d_a_tilde / sum(d_a_tilde)  # document i weighted words, normalized

        neighbors_ids = np.argsort(D_c[:, i])  # sort by the order of the distance to the document i
        neighbors_ids = neighbors_ids[1:n_neighbours + 1]  # use only N_size nearest neighbors ids

        # Compute WMD from xi to the rest documents
        dD_dA_all = dict()
        alpha_all = dict()
        beta_all = dict()

        # 'nn' for nearest neighbors (Eucledian)
        neighbors = dataloader.batch_for_indices(neighbors_ids)

        # TODO: fix multiprocessing
        # pool = mul.Pool(processes = 6)
        sinkhorn_results = []

        for j in range(0, n_neighbours):
            # Computing smoothed WMD

            xj, bow_j, ids_j, yj = dataloader[neighbors[j]]
            # M = distance(np.dot(A,xi), np.dot(A,xj))
            ids_j.shape = ids_j.size
            bow_j.shape = (np.size(bow_j), 1)
            d_b_tilde = bow_j * w[ids_j]
            d_b_tilde = d_b_tilde / sum(d_b_tilde)  # 'other' document, weighted words, normalized

            # RES.append(pool.apply_async(sinkhorn3, (ii, j, A, xi, xj, a, b)))

            # WARNING: sinkhorn2 and sinkhorn3 have different sets of return parameters
            sinkhorn_results.append(sinkhorn3(A, xi, xj, d_a_tilde, d_b_tilde))

        # pool.close()
        # pool.join()

        Di = np.zeros([n_neighbours, 1])  # (squared??) sinkhorn distances - relaxed WMD distances
        for j, res in enumerate(sinkhorn_results):
            # r = res.get()
            r = res
            Di[j] = r[3]
            alpha_all[j] = r[0]
            beta_all[j] = r[1]
            transport_matrix = r[2]  # T_ij in the paper
            xi = r[4]
            xj = r[5]
            d_a_tilde = r[6]
            d_b_tilde = r[7]
            d_a_tilde.shape = (np.size(d_a_tilde),)
            d_b_tilde.shape = (np.size(d_b_tilde),)

            # gradient for metric
            dD_dA_all[j] = np.dot(xi * d_a_tilde.T, xi.T) \
                           + np.dot(xj * d_b_tilde.T, xj.T) \
                           - np.dot(np.dot(xi, transport_matrix), xj.T) \
                           - np.dot(np.dot(xj, transport_matrix.T), xi.T)

        # Compute NCA probabilities
        Di[Di < 0] = 0
        dmin = min(Di) # WTF: why use dmin?
        p_i = np.exp(-Di + dmin) + EPS  # WTF: why eps in the nominator?
        p_i = p_i / sum(p_i)
        y_mask = [y_nn == yi for _, _, _, y_nn in neighbors]  # all neighbors with the same class as d_a
        p_a = sum(p_i[y_mask])  # probability for document d_a
        p_a = p_a + EPS  # to avoid division by 0

        # Compute gradient wrt w and A
        # logging.debug('\tComputing w and A gradients for batch')
        dw_ii = np.zeros(np.size(w))
        dA_ii = np.zeros([dim, dim])

        for j in range(0, n_neighbours):
            # print('\tIter {}'.format(j))
            _, bow_j, ids_j, yj = dataloader[neighbors[j]]

            c_ij = p_i[j] / p_a * int(yj == yi) - p_i[j]
            # ids_j.shape = ids_j.size
            d_b_tilde = bow_j * w[ids_j]  # there was a transposition, but it should not be here (I suppose)
            d_b_tilde = d_b_tilde / sum(d_b_tilde)
            d_a_sum = sum(w[ids_i] * bow_i)
            d_b_sum = sum(w[ids_j] * bow_j)

            # dD_Aw / dw:
            dwmd_dwi = bow_i * alpha_all[j] / d_a_sum - bow_i * (np.dot(alpha_all[j].T, d_a_tilde) / d_a_sum)
            dwmd_dwj = bow_j * beta_all[j] / d_b_sum - bow_j * (np.dot(beta_all[j].T, d_b_tilde) / d_b_sum)

            dw_ii[ids_i] = dw_ii[ids_i] + (c_ij * dwmd_dwi).squeeze(1)
            dw_ii[ids_j] = dw_ii[ids_j] + (c_ij * dwmd_dwj).squeeze(1)

            dA_ii = dA_ii + c_ij * dD_dA_all[j]

        if sum(np.isnan(dw_ii)) == 0 and sum(sum(np.isnan(dA_ii))) == 0:
            dw_ii.shape = [np.size(w), 1]
            dw = dw + dw_ii
            dA_aux = dA_aux + dA_ii
            tr_loss = tr_loss - np.log(p_a)
        else:
            n_nan = n_nan + 1

    dA = np.dot(A, dA_aux)

    batch_size = batch_size - n_nan
    if n_nan > 0:
        logging.info('number of bad samples: ' + str(n_nan))
        assert n_nan < batch_size, 'every gradient in the batch is NaN'

    tr_loss = tr_loss / batch_size

    dw = dw / batch_size
    dA = dA / batch_size

    del D_c, xi, yj, Di, d_a_tilde, d_b_tilde, dw_ii, dA_ii, dwmd_dwi, dwmd_dwj
    gc.collect()

    return dw, dA


def sinkhorn2(np.ndarray[np.double_t, ndim =2] A,
              np.ndarray[np.double_t, ndim =2] xi,
              np.ndarray[np.double_t, ndim =2] xj,
              np.ndarray[np.double_t, ndim =2] a,
              np.ndarray[np.double_t, ndim =2] b):
    # https://arxiv.org/pdf/1306.0895.pdf

    cdef float change, obj_primal
    cdef np.ndarray[np.double_t, ndim =2] M, K, Kt, u, u0, v, alpha, beta, T
    cdef np.ndarray[np.double_t, ndim =1] z
    cdef int l, iteR

    M = distance(np.dot(A, xi.T), np.dot(A, xj.T))
    M[M<0] = 0
    
    l = len(a)
    K = np.exp(-cfg.sinkhorn.lambda_ * M)
    Kt = K / a
    u = np.ones([l,1]) /l
    iteR = 0
    change = np.inf

    while change > cfg.sinkhorn.float_tol and iteR <= cfg.sinkhorn.max_iter:
        iteR = iteR + 1
        u0 = u
        # sinkhorn distance formula:
        u = 1.0 / (np.dot(Kt, (b / (np.dot(K.T, u)))))
        change = np.linalg.norm(u - u0) / np.linalg.norm(u)

    if min(u) <= 0:
        u = u - min(u) + EPS

    v = b / (np.dot(K.T, u))

    if min(v) <= 0:
        v = v - min(v) + EPS

    alpha = np.log(u)
    alpha = 1.0 / cfg.sinkhorn.lambda_ * (alpha - np.mean(alpha))
    beta = np.log(v)
    beta = 1.0 / cfg.sinkhorn.lambda_ * (beta - np.mean(beta))
    #v.shape = (np.size(v),)
    z = v.T[0]
    T = z * (K * u)
    obj_primal = np.sum(T*M)#sum(sum(T*M))
    # obj_dual = a * alpha + b * beta
  
    return alpha, beta, T, obj_primal


def sinkhorn3(np.ndarray[np.double_t, ndim =2] A,
              np.ndarray[np.double_t, ndim =2] xi,
              np.ndarray[np.double_t, ndim =2] xj,
              np.ndarray[np.double_t, ndim =2] a,
              np.ndarray[np.double_t, ndim =2] b):
    # https://arxiv.org/pdf/1306.0895.pdf

    cdef float change, obj_primal
    cdef np.ndarray[np.double_t, ndim =2] M, K, Kt, u, u0, v, alpha, beta, T
    cdef np.ndarray[np.double_t, ndim =1] z
    cdef int l, iteR

    M = distance(np.dot(A, xi), np.dot(A, xj))
    M[M<0] = 0
    
    l = len(a)
    K = np.exp(-cfg.sinkhorn.lambda_ * M)
    Kt = K / a
    u = np.ones([l, 1]) / l
    iteR = 0
    change = np.inf
    #b.shape = (np.size(b),1)

    while change > cfg.sinkhorn.float_tol and iteR <= cfg.sinkhorn.max_iter:
        iteR = iteR + 1
        u0 = u
        # sinkhorn distance formula:
        u = 1.0 / (np.dot(Kt, (b / (np.dot(K.T, u)))))
        change = np.linalg.norm(u - u0) / np.linalg.norm(u)

    if min(u) <= 0:
        u = u - min(u) + EPS

    v = b / (np.dot(K.T, u))

    if min(v) <= 0:
        v = v - min(v) + EPS

    alpha = np.log(u)
    alpha = 1.0 / cfg.sinkhorn.lambda_ * (alpha - np.mean(alpha))
    beta = np.log(v)
    beta = 1.0 / cfg.sinkhorn.lambda_ * (beta - np.mean(beta))
    # v.shape = (np.size(v),)
    z = v.T[0]
    T = z * (K * u)
    obj_primal = np.sum(T*M) # sum(sum(T*M))
    # obj_dual = a * alpha + b * beta

    return alpha, beta, T, obj_primal, xi, xj, a, b


def knn_fall_back(DE, y_train, y_test, k_neighbors_list):
    """
    Computes KNN error rate
    :param DE: distance matrix, DE[i,j] = distance(train[i], test[j])
    """
    # TODO: try to use scipy or change to faster knn approximation algorithm
    k_neighbors_list = sorted(k_neighbors_list)

    n_test = DE.shape[1]
    _, ix = mink(DE, k_neighbors_list[-1])

    predictions = np.zeros([len(k_neighbors_list), n_test])

    for i in range(0, len(k_neighbors_list)):
        still_voting = np.ones(n_test)
        k = k_neighbors_list[i]
        # TODO: change to for-clause
        while 1:
            topk_indices = ix[0:k, :]
            try:
                logging.debug('ix.shape: %s' % str(ix.shape))
                logging.debug(topk_indices)
                logging.debug('topk_indices type: %s' % type(topk_indices))
            except Exception as e:
                logging.error('Error in knn_fall_back logging')
                logging.error(e)

            sam = y_train[topk_indices]
            vote, count = stats.mode(sam)
            vote = vote[0]
            count = count[0]

            not_sure = count < k / 2
            if np.sum(still_voting * not_sure) == 0:
                uneq = still_voting != 0
                predictions[k, uneq] = vote[uneq]
                if np.sum(predictions[k, :] == 0) != 0:
                    logging.error("unknown error in knn_fall_back")
                break

            conf = still_voting - not_sure
            conf = conf == 1

            predictions[i, conf] = vote[conf]

            still_voting = still_voting * not_sure
            if k == 1:
                uneq = still_voting != 0
                predictions[i, uneq] = vote[uneq]
                if np.sum(predictions[i, :] == 0) != 0:
                    logging.error("unknown error in knn_fall_back")
                break
            k = k - 2

    # NOTE: can be speeded up using matrix computation (?)
    err = np.ones(len(k_neighbors_list))
    for i in range(0, len(k_neighbors_list)):
        err[i] = np.mean(predictions[i, :] != y_test)

    return err


def mink(M, k):
    # NOTE: use only argsort, index for getting values to speed up a bit
    sortM = np.sort(M, 0)
    idM = np.argsort(M, 0)
    sortM = sortM[0:k, :]
    idM = idM[0:k, :]
    return sortM, idM
