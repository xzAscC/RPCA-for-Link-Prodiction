import numpy as np
import random
from numpy.linalg import norm, svd


def divideNetwork(Networks, ratio):
    train = np.matrix(Networks)
    row, col = np.nonzero(np.tril(Networks))
    probe_size = round(row.shape[0] * (1 - ratio))
    for i in range(probe_size):
        rand_number = int(row.shape[0] * random.random())
        train[row[rand_number], col[rand_number]] = 0
        train[col[rand_number], row[rand_number]] = 0
    test = Networks - train
    # # if (test == np.zeros((Networks.shape[0], Networks.shape[1]))).all():
    # # if (Networks == train).all():
    # if((test + train == Networks).all()):
    #     print(1)
    return train, test


def alm_rpca(X, lmbda, maxiter=100, tol=1e-3,):
    Y = X
    norm_two = norm(Y.ravel(), 2)
    norm_inf = norm(Y.ravel(), np.inf) / lmbda
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y / dual_norm
    A = np.zeros(Y.shape)
    E = np.zeros(Y.shape)
    d_norm = norm(X, 'fro')
    mu = 1.25 / norm_two
    rho = 1.5
    sv = 10.
    n = Y.shape[0]
    itr = 0
    while True:
        tmp = X - A + (1 / mu) * Y
        E_update = np.maximum(tmp - lmbda / mu, 0) + np.minimum(tmp + lmbda / mu, 0)
        U, S, V = svd(X - E_update + (1 / mu) * Y, full_matrices=False)
        svp = (S > 1 / mu).shape[0]
        if svp < sv:
            sv = np.min([svp + 1, n])
        else:
            sv = np.min([svp + round(.05 * n), n])
        A_update = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
        A = A_update
        E = E_update
        Z = X - A - E
        Y = Y + mu * Z
        mu = np.min([mu * rho, mu * 1e7])
        itr += 1
        if ((norm(Z, 'fro') / d_norm) < tol) or (itr >= maxiter):
            break
    return A, E


def compute_precision(NetworkPrediction, train, test):
    precision = 0
    train = np.tril(train)
    NetworksP = np.tril(NetworkPrediction - np.multiply(np.eye(NetworkPrediction.shape[0]), NetworkPrediction))
    row, col = np.nonzero(np.tril(test))
    probe_size1 = row.shape[0]
    NetworksP *= np.where(train != 0, 0, 1)
    row, col = np.nonzero(NetworksP)
    probe_size2 = row.shape[0]
    rand_number = probe_size1 if probe_size1 < probe_size2 else probe_size2
    # print(rand_number)
    # print(probe_size1)
    # print(probe_size2)
    for i in range(rand_number):
        row, col = np.unravel_index(np.argmax(NetworksP), NetworksP.shape)
        NetworksP[row, col] = 0
        if test[row, col] > 0:
            precision += 1
    precision /= probe_size1
    return precision
