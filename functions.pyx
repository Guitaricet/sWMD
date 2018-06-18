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
import scipy as sp

import scipy.io as sio
import scipy.spatial.distance as sdist

from scipy import stats

import pyximport
pyximport.install(reload_support=True)

cimport numpy as np
cimport cython


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')


MAX_ITER = 200  # for sinkhorn distance
FLOAT_TOL = 1e-3
EPSILON = 1e-8


def distance(np.ndarray[np.double_t, ndim =2] X, np.ndarray[np.double_t, ndim =2] x):
    # computes the pairwise squared distance matrix
    # between any column vectors in X and in x

    cdef int D, N, d, n
    cdef np.ndarray[np.double_t, ndim =2] dist

    D = np.size(X[:,0])
    N = np.size(X[0,:])
    d = np.size(x[:,0])
    n = np.size(x[0,:])
    if D != d:
        logging.error('Both sets of vectors must have same dimensionality!')
        os._exit()
    dist = sdist.cdist(X.T, x.T, 'sqeuclidean')

    return dist


def grad_swmd(dataloader, document_centers, w, A, lambda_, batch_size, n_neighbours):
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
    :param lambda_: smoothing parameter (entropy regularisation weight) — higher labmda_ closer sinkhorn to WMD
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

            # RES.append(pool.apply_async(sinkhorn3, (ii, j, A, xi, xj, a, b, lambda_, 200, 1e-3)))

            # WARNING: sinkhorn2 and sinkhorn3 have different sets of return parameters
            sinkhorn_results.append(
                sinkhorn3(A, xi, xj, d_a_tilde, d_b_tilde, lambda_, MAX_ITER, FLOAT_TOL)
            )

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
        p_i = np.exp(-Di + dmin) + EPSILON  # WTF: why eps in the nominator?
        p_i = p_i / sum(p_i)
        y_mask = [y_nn == yi for _, _, _, y_nn in neighbors]  # all neighbors with the same class as d_a
        p_a = sum(p_i[y_mask])  # probability for document d_a
        p_a = p_a + EPSILON  # to avoid division by 0

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
              np.ndarray[np.double_t, ndim =2] b,
              int lambdA,
              int max_iter,
              float tol):
    # https://arxiv.org/pdf/1306.0895.pdf

    cdef float epsilon, change, obj_primal
    cdef np.ndarray[np.double_t, ndim =2] M, K, Kt, u, u0, v, alpha, beta, T
    cdef np.ndarray[np.double_t, ndim =1] z
    cdef int l, iteR

    epsilon = 1e-10
    logging.debug('A.shape: %s, xi.shape: %s, xj.shape: %s' % (str(A.shape), str(xi.shape), str(xj.shape)))
    M = distance(np.dot(A, xi.T), np.dot(A, xj.T))
    M[M<0] = 0
    
    l = len(a)
    K = np.exp(-lambdA * M)
    Kt = K / a
    u = np.ones([l,1]) /l
    iteR = 0
    change = np.inf

    while change > tol and iteR <= max_iter:
        iteR = iteR + 1
        u0 = u
        # sinkhorn distance formula:
        u = 1.0 / (np.dot(Kt, (b / (np.dot(K.T, u)))))
        change = np.linalg.norm(u - u0) / np.linalg.norm(u)

    if min(u) <= 0:
        u = u - min(u) + epsilon

    v = b / (np.dot(K.T, u))

    if min(v) <= 0:
        v = v - min(v) + epsilon

    alpha = np.log(u)
    alpha = 1.0 / lambdA * (alpha - np.mean(alpha))
    beta = np.log(v)
    beta = 1.0 / lambdA * (beta - np.mean(beta))
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
              np.ndarray[np.double_t, ndim =2] b,
              int lambdA,
              int max_iter,
              float tol):
    # https://arxiv.org/pdf/1306.0895.pdf

    cdef float epsilon, change, obj_primal
    cdef np.ndarray[np.double_t, ndim =2] M, K, Kt, u, u0, v, alpha, beta, T
    cdef np.ndarray[np.double_t, ndim =1] z
    cdef int l, iteR


    epsilon = 1e-10

    M = distance(np.dot(A, xi), np.dot(A, xj))
    M[M<0] = 0
    
    l = len(a)
    K = np.exp(-lambdA * M)
    Kt = K / a
    u = np.ones([l, 1]) / l
    iteR = 0
    change = np.inf
    #b.shape = (np.size(b),1)

    while change > tol and iteR <= max_iter:
        iteR = iteR + 1
        u0 = u
        # sinkhorn distance formula:
        u = 1.0 / (np.dot(Kt, (b / (np.dot(K.T, u)))))
        change = np.linalg.norm(u - u0) / np.linalg.norm(u)

    if min(u) <= 0:
        u = u - min(u) + epsilon

    v = b / (np.dot(K.T, u))

    if min(v) <= 0:
        v = v - min(v) + epsilon

    alpha = np.log(u)
    alpha = 1.0 / lambdA * (alpha - np.mean(alpha))
    beta = np.log(v)
    beta = 1.0 / lambdA * (beta - np.mean(beta))
    # v.shape = (np.size(v),)
    z = v.T[0]
    T = z * (K * u)
    obj_primal = np.sum(T*M) # sum(sum(T*M))
    # obj_dual = a * alpha + b * beta

    return alpha, beta, T, obj_primal, xi, xj, a, b


def knn_swmd(dataloader_train, dataloader_test, w, lambda_, A):
    """
    Computes KNN
    Not time and memory efficient, because computes full distance matrix

    :param A, w: model parameters
    :param lambda_: WMD relaxation parameter
    :return: KNN error rate
    """
    n_train = len(dataloader_train)
    n_test = len(dataloader_test)

    wmd_dist = np.zeros([n_train, n_test])

    # TODO: fix multiprocessing
    # pool = mul.Pool(processes = 6)
    result = []

    for i in range(0, n_train):

        Wi = np.zeros(n_test)
        x_i, bow_i, indices_train_i, _ = dataloader_train[i]

        bow_i.shape = [np.size(bow_i), 1]
        logging.debug('w.shape: %s' % str(w.shape))
        logging.debug('indices_train_i.shape: %s' % str(indices_train_i.shape))
        logging.debug('w[indices_train_i][0]: %s' % str(w[indices_train_i][0]))
        d_a = bow_i * w[indices_train_i]
        d_a = d_a / sum(d_a)

        for j in range(0, n_test):
            x_j, bow_j, indices_test_j, _ = dataloader_test[j]
            bow_j.shape = [np.size(bow_j), 1]
            d_b = bow_j * w[indices_test_j][0]
            d_b = d_b / sum(d_b)
            d_b.shape = (np.size(d_b), 1)

            # result.append(pool.apply_async(sinkhorn2, (i, j, A, x_i, x_j, a, b, lambda_, 200, 1e-3)))

            # WARNING! sinkhorn2 and sinkhorn3 have different sets of return parameters
            result.append(sinkhorn2(A, x_i, x_j, d_a, d_b, lambda_, 200, 1e-3))
            # print("n_train {} n_test {} is finished".format(i, j))

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

    err = knn_fall_back(wmd_dist, dataloader_train.labels, dataloader_test.labels, [5])

    del wmd_dist
    gc.collect()

    return err


def knn_fall_back(DE, y_train, y_test, k_neighbors_list):
    """
    Computes KNN error rate
    """
    [n, ne] = [np.size(DE, 0), np.size(DE, 1)]
    [dists, ix] = mink(DE, k_neighbors_list[-1])

    pe = np.zeros([len(k_neighbors_list), ne])

    for k in range(0, len(k_neighbors_list)):
        still_voting = np.ones(ne)
        kcopy = k_neighbors_list[k]
        while 1:
            sam = y_train[ix[0:kcopy, :]]
            [vote, count] = stats.mode(sam)
            vote = vote[0]
            count = count[0]

            not_sure = count < kcopy / 2
            if np.sum(still_voting * not_sure) == 0:
                uneq = still_voting != 0
                pe[k, uneq] = vote[uneq]
                if np.sum(pe[k, :] == 0) != 0:
                    logging.error("unknown error in knn_fall_back")
                break

            conf = still_voting - not_sure
            conf = conf == 1

            pe[k, conf] = vote[conf]

            still_voting = still_voting * not_sure
            if kcopy == 1:
                uneq = still_voting != 0
                pe[k, uneq] = vote[uneq]
                if np.sum(pe[k, :] == 0) != 0:
                    logging.error("unknown error in knn_fall_back")
                break
            kcopy = kcopy - 2

    err = np.ones(len(k_neighbors_list))
    for k in range(0, len(k_neighbors_list)):
        err[k] = np.mean(pe[k, :] != y_test)

    return err


def mink(M, k):
    sortM = np.sort(M,0)
    idM = np.argsort(M,0)
    sortM = sortM[0:k,:]
    idM = idM[0:k,:]
    return sortM, idM
