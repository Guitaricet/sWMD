##################################
###### Writen by Boyuan Pan ######
##################################

import os
import gc
import random
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


MAX_ITER = 200
FLOAT_TOL = 1e-3


def load_data(dataset, seed):
    if dataset == 'ohsumed' or dataset == 'r83' or dataset == '20ng2' or dataset == '20ng2_500':
        data = 'dataset/' + dataset + '_tr_te.mat'
        data = sio.loadmat(data)
    else:
        data = 'dataset/' + dataset + '_tr_te_split.mat'
        data = sio.loadmat(data)
        x_train = data['X'][0][data['TR'][seed,:]-1]
        x_test = data['X'][0][data['TE'][seed,:]-1]
        BOW_x_train = data['BOW_X'][0][data['TR'][seed,:]-1]
        BOW_x_test = data['BOW_X'][0][data['TE'][seed,:]-1]
        indices_train = data['indices'][0][data['TR'][seed,:]-1]
        indices_test = data['indices'][0][data['TE'][seed,:]-1]
        y_train = data['Y'][0][data['TR'][seed,:]-1]
        y_test = data['Y'][0][data['TE'][seed,:]-1]
        return x_train, x_test, y_train, y_test, BOW_x_train, BOW_x_test, indices_train, indices_test

def minclass(y, idx):
    if len(idx) == 0:
        return 0

    un = np.unique(y)
    m = float('inf')
    for i in range(0,len(un)):
        m = min(sum(y[idx] == un[i]), m)
    return m


def makesplits(y, split, splits, classsplit=0, k=1):
    # SPLITS "y" into "splits" sets with a "split" ratio.
    # if classsplit==1 then it takes a "split" fraction from each class

    if split == 1:
        train = np.array(range(len(y)))
        random.shuffle(train)
        test = []
        return train, test

    if split == 0:
        test = np.array(range(len(y)))
        random.shuffle(test)
        train = []
        return train, test

    n = len(y)
    # TODO: debug this
    # if minclass(y, np.array(range(0,len(y)))) < k or split * len(y) / len(np.unique(y)) < k :
    #     print 'K:'+ k + ' split:' + split + ' n:' + len(y)
    #     print 'Cannot sub-sample splits! Reduce number of neighbors_ids.'
    #     os._exit()


    if classsplit:
        un = np.unique(y)
        for i in range(0, splits):
            trsplit = []
            tesplit = []
            print(trsplit)
            while minclass(y, trsplit) < k:
                for j in range(0,len(un)):
                    ii = np.where(y == un[j])
                    ii = np.array(ii[0])
                    co = int(round(split * np.size(ii)))
                    random.shuffle(ii)
                    trsplit = np.append(trsplit, ii[0:co])
                    tesplit = np.append(tesplit, ii[co:np.size(ii)])

                    trsplit = np.array(map(int,trsplit))
                    tesplit = np.array(map(int,tesplit))

            train = trsplit
            test = tesplit

    else:
        for i in range(0, splits):
            trsplit = []
            tesplit = []
            while minclass(y,trsplit) < k:
                ii = np.array(range(n))
                random.shuffle(ii)
                co = int(round(split*n))
                trsplit = ii[0:co]
                tesplit = ii[co:n]

            train = trsplit
            test = tesplit

    return train, test


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
        print('Both sets of vectors must have same dimensionality!')
        os._exit()
    dist = sdist.cdist(X.T, x.T, 'sqeuclidean')

    return dist

def grad_swmd(x_train, y_train, bow_x_train, indices_train, xtr_center, w, A, lambda_, batch, n_neighbours):
    """
    Computes gradients with respect to A and w
    The formula can be found in https://papers.nips.cc/paper/6139-supervised-word-movers-distance.pdf
    (formula 8 and 10, page 4)

    :param x_train: matrix of documents (embed_dim x doc_idx x n_words)
    :param y_train: list of classes of documents
    :param bow_x_train: bag-of-words text representations (d in the paper)
    :param indices_train:
    :param xtr_center:
    :param w: words weights (learnable)
    :param A: transformation matrix
    :param lambda_: smoothing parameter (entropy regularisation weight) — higher labmda_ closer sinkhorn to WMD
    :param batch: batch of documents
    :param n_neighbours: number of nearest neighbours
    :return: dw, dA - values of gradients
    """

    print('In grad_swmd')

    epsilon = 1e-8

    dim = np.size(x_train[0], 0)  # dimension of word vector
    n_train = len(y_train)  # number of documents

    dw = np.zeros([np.size(w), 1])
    dA = np.zeros([dim, dim])

    # Sample documents
    sample_doc_idx = random.sample(range(n_train), batch)

    # Euclidean distances between document centers
    D_c = distance(np.dot(A, xtr_center), np.dot(A, xtr_center))
    tr_loss = 0
    n_nan = 0

    print('Starting batch iteration')

    for batch_idx in range(0, batch):
        print('Batch {}'.format(batch_idx))

        doc_idx = sample_doc_idx[batch_idx]
        xi = x_train[doc_idx]
        yi = y_train[doc_idx]
        idx_i = indices_train[doc_idx]
        idx_i.shape = idx_i.size
        bow_i = bow_x_train[doc_idx]
        bow_i.shape = (np.size(bow_i), 1)
        d_a_tilde = bow_i * w[idx_i]
        d_a_tilde = d_a_tilde / sum(d_a_tilde)  # 'our' document, weighted words, normalized

        neighbors_ids = np.argsort(D_c[:, doc_idx])  # sort by the order of the distance to the 'i' document

        # Compute WMD from xi to the rest documents
        neighbors_ids = neighbors_ids[1:n_neighbours + 1]  # use only N_size nearest neighbors ids
        dD_dA_all = dict()
        alpha_all = dict()
        beta_all = dict()

        x_train_nn = x_train[neighbors_ids]  # 'nn' for nearest neighbors (Eucledian)
        y_train_nn = y_train[neighbors_ids]
        bow_x_train_nn = bow_x_train[neighbors_ids]
        indices_train_nn = indices_train[neighbors_ids]

        # TODO: fix multiprocessing
        # pool = mul.Pool(processes = 6)
        sinkhorn_results = []

        for j in range(0, n_neighbours):
            # Computing smoothed WMD

            xj = x_train_nn[j]
            yj = y_train_nn[j]
            # M = distance(np.dot(A,xi), np.dot(A,xj))
            ids_j = indices_train_nn[j]
            ids_j.shape = ids_j.size
            bow_j = bow_x_train_nn[j]
            bow_j.shape = (np.size(bow_j), 1)
            d_b_tilde = bow_j * w[ids_j]
            d_b_tilde = d_b_tilde / sum(d_b_tilde)  # 'other' document, weighted words, normalized

            # RES.append(pool.apply_async(sinkhorn3, (ii, j, A, xi, xj, a, b, lambda_, 200, 1e-3)))

            # WARNING! sinkhorn2 and sinkhorn3 have different sets of return parameters
            sinkhorn_results.append(
                sinkhorn3(batch_idx, j, A, xi, xj, d_a_tilde, d_b_tilde, lambda_, MAX_ITER, FLOAT_TOL)
            )

        # pool.close()
        # pool.join()

        Di = np.zeros([n_neighbours, 1])  # (squared??) sinkhorn distances (relaxed WMD distances)
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
        dmin = min(Di)
        p_i = np.exp(-Di + dmin) + epsilon
        p_i[y_train_nn == doc_idx] = 0
        p_i = p_i / sum(p_i)
        p_a = sum(p_i[y_train_nn == yi])  # probability for document d_a
        p_a = p_a + epsilon  # to avoid division by 0

        # Compute gradient wrt w and A
        print('\tComputing w and A gradients for batch')
        dw_ii = np.zeros(np.size(w))
        dA_ii = np.zeros([dim, dim])

        for j in range(0, n_neighbours):
            # print('\tIter {}'.format(j))

            yi_bin_mask = y_train_nn[j] == yi
            c_ij = p_i[j] / p_a * yi_bin_mask - p_i[j]
            ids_j = indices_train_nn[j]
            ids_j.shape = ids_j.size
            bow_j = bow_x_train_nn[j]
            d_b_tilde = bow_j * w[ids_j]  # there was a transposition, but it should not be here (I suppose)
            d_b_tilde = d_b_tilde / sum(d_b_tilde)
            d_a_sum = sum(w[idx_i] * bow_i)
            d_b_sum = sum(w[ids_j] * bow_j)

            # dD_Aw / dw:
            dwmd_dwi = bow_i * alpha_all[j] / d_a_sum - bow_i * (np.dot(alpha_all[j].T, d_a_tilde) / d_a_sum)
            dwmd_dwj = bow_j * beta_all[j] / d_b_sum - bow_j * (np.dot(beta_all[j].T, d_b_tilde) / d_b_sum)

            # print('dw_ii: ', dw_ii.shape)
            # print('dw_ii[0]: ', dw_ii[0].shape)
            # print('idx_i: ', idx_i)
            # print('dw_ii[idx_i]: ', dw_ii[idx_i].shape)
            # print(dw_ii[idx_i])
            # print('cij: ', cij)
            # print('dwmd_dwi: ', dwmd_dwi.shape)
            # print('dwmd_dwi: ', dwmd_dwi)
            # print('cij * dwmd_dwi', (cij * dwmd_dwi).shape)
            # value = dw_ii[idx_i] + (cij * dwmd_dwi).squeeze(1)
            # print('value:', value.shape)

            # TODO: crutch
            dw_ii[idx_i] = dw_ii[idx_i] + (c_ij * dwmd_dwi).squeeze(1)
            dw_ii[ids_j] = dw_ii[ids_j] + (c_ij * dwmd_dwj).squeeze(1)

            dA_ii = dA_ii + c_ij * dD_dA_all[j]

        if sum(np.isnan(dw_ii)) == 0 and sum(sum(np.isnan(dA_ii))) == 0:
            dw_ii.shape = [np.size(w), 1]
            dw = dw + dw_ii
            dA = dA + dA_ii
            tr_loss = tr_loss - np.log(p_a)
        else:
            n_nan = n_nan + 1

    dA = np.dot(A, dA)

    batch = batch - n_nan
    if n_nan > 0:
        print('number of bad samples: ' + str(n_nan))

    tr_loss = tr_loss / batch

    dw = dw / batch
    dA = dA / batch

    del D_c, xi, yj, Di, d_a_tilde, d_b_tilde, dw_ii, dA_ii, dwmd_dwi, dwmd_dwj
    gc.collect()

    return dw, dA


def sinkhorn2(int i, int j,  np.ndarray[np.double_t, ndim =2] A, np.ndarray[np.double_t, ndim =2] xi,  np.ndarray[np.double_t, ndim =2] xj, np.ndarray[np.double_t, ndim =2] a,  np.ndarray[np.double_t, ndim =2] b, int lambdA, int max_iter, float tol):
    # https://arxiv.org/pdf/1306.0895.pdf

    cdef float epsilon, change, obj_primal
    cdef np.ndarray[np.double_t, ndim =2] M, K, Kt, u, u0, v, alpha, beta, T
    cdef np.ndarray[np.double_t, ndim =1] z
    cdef int l, iteR

    epsilon = 1e-10

    M = distance(np.dot(A,xi), np.dot(A,xj))
    M[M<0] = 0
    
    l = len(a)
    K = np.exp(-lambdA * M)
    Kt = K/a
    u = np.ones([l,1]) /l
    iteR = 0
    change = np.inf

    while change > tol and iteR <= max_iter:
        iteR = iteR + 1
        u0 = u
        u = 1.0/(np.dot(Kt,(b/(np.dot(K.T,u)))))

        change = np.linalg.norm(u - u0) / np.linalg.norm(u)

    if min(u) <= 0:
        u = u - min(u) + epsilon

    v = b/(np.dot(K.T,u))

    if min(v) <= 0:
        v = v - min(v) + epsilon

    alpha = np.log(u)
    alpha = 1.0/lambdA * (alpha - np.mean(alpha))
    beta = np.log(v)
    beta = 1.0/lambdA * (beta - np.mean(beta))
    #v.shape = (np.size(v),)
    z = v.T[0]
    T = z * (K * u)
    obj_primal = np.sum(T*M)#sum(sum(T*M))
    # obj_dual = a * alpha + b * beta
  
    # print "ntr "+ str(i) + " nte " + str(j) + " is finished"
    return alpha, beta, T, obj_primal


def sinkhorn3(int i, int j,
              np.ndarray[np.double_t, ndim =2] A,
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

    M = distance(np.dot(A,xi), np.dot(A,xj))
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
    #v.shape = (np.size(v),)
    z = v.T[0]
    T = z * (K * u)
    obj_primal = np.sum(T*M) # sum(sum(T*M))
  #  obj_dual = a * alpha + b * beta

    return alpha, beta, T, obj_primal, xi, xj, a, b


def knn_swmd(x_train, y_train, x_test, y_test, BOW_x_train, BOW_x_test, indices_train, indices_test, w, lambda_, A):
    n_train = len(y_train)
    n_test = len(y_test)

    WMD = np.zeros([n_train, n_test])

    # TODO: fix multiprocessing
    # pool = mul.Pool(processes = 6)
    result = []

    for i in range(0,n_train):
        
        Wi = np.zeros(n_test)
        xi = x_train[i]
        bow_i = BOW_x_train[i]
        bow_i.shape = [np.size(bow_i),1]
        a = bow_i * w[indices_train[i]][0]
        a = a / sum(a)
        

        for j in range(0,n_test):
            xj = x_test[j]
            bow_j = BOW_x_test[j]
            bow_j.shape = [np.size(bow_j),1]
            b = bow_j * w[indices_test[j]][0]
            b = b / sum(b)
            b.shape = (np.size(b),1)

            # result.append(pool.apply_async(sinkhorn2, (i, j, A, xi, xj, a, b, lambda_, 200, 1e-3)))

            # WARNING! sinkhorn2 and sinkhorn3 have different number of return parameters
            result.append(sinkhorn2(i, j, A, xi, xj, a, b, lambda_, 200, 1e-3))
            # print("n_train {} n_test {} is finished".format(i, j))

    # pool.close()
    # pool.join()

    n = 0
    for res in result:
        # r = res.get()
        r = res
        i = n / n_test
        j = np.mod(n,n_test)
        WMD[i,j] = r[3]
        n+=1

    err = knn_fall_back(WMD, y_train, y_test, range(1,20))

    del WMD
    gc.collect()

    return err

def knn_fall_back(DE, y_train, y_test, ks):
    [n,ne] = [np.size(DE,0), np.size(DE,1)]
    [dists, ix] = mink(DE,ks[-1])

    pe = np.zeros([len(ks),ne])

    for k in range(0,len(ks)):
        still_voting = np.ones(ne)
        kcopy = ks[k]
        while 1:
            sam = y_train[ix[0:kcopy,:]]
            [vote,count]= stats.mode(sam)
            vote = vote[0]
            count = count[0]

            not_sure = count < kcopy/2
            if np.sum(still_voting * not_sure) == 0:
                uneq = still_voting != 0
                pe[k,uneq] = vote[uneq]
                if np.sum(pe[k,:] == 0) != 0:
                    print("error")
                break

            conf = still_voting - not_sure
            conf = conf == 1

            pe[k,conf] = vote[conf]

            still_voting = still_voting * not_sure
            if kcopy == 1:
                uneq = still_voting != 0
                pe[k,uneq] = vote[uneq]
                if np.sum(pe[k,:] == 0) != 0:
                    print("error")
                break
            kcopy = kcopy - 2

    err = np.ones(len(ks))
    for k in range(0,len(ks)):
        err[k] = np.mean(pe[k,:] != y_test)

    return err

def mink(M,k):
    sortM = np.sort(M,0)
    idM = np.argsort(M,0)
    sortM = sortM[0:k,:]
    idM = idM[0:k,:]
    return sortM, idM
