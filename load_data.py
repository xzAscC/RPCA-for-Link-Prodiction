import numpy as np
from scipy.sparse import coo_matrix


def loadData(choose):
    if choose == "1":
        arr = np.loadtxt('dataset/Jazz.txt', dtype='int')
        data = arr[:, 2] - 0.5
        row = arr[:, 0] - 1
        col = arr[:, 1] - 1
        ind = max(col) + 1
        OriginalNetwork = coo_matrix((data, (row, col)), shape=(ind, ind)).toarray()
        OriginalNetwork += OriginalNetwork.T
        Odiag = np.diagonal(OriginalNetwork)
        if max(Odiag) != 0:
            OriginalNetwork -= np.diag(Odiag)
        return OriginalNetwork
        # 获取数据和值不为0的矩阵索引
    if choose == "2":
        arr = np.loadtxt('dataset/contact.txt', dtype='int')
        row = arr[:, 0]
        col = arr[:, 1]
        data = [1] * row.shape[0]
        ind = max(col) + 1
        OriginalNetwork = coo_matrix((data, (row, col)), shape=(ind, ind)).toarray()
        OriginalNetwork += OriginalNetwork.T
        Odiag = np.diagonal(OriginalNetwork)
        if max(Odiag) != 0:
            OriginalNetwork -= np.diag(Odiag)
        return OriginalNetwork
    if choose == "3":
        arr = np.loadtxt('dataset/PoliticalBlogs.txt', dtype='int')
        row = arr[:, 0] - 1
        col = arr[:, 1] - 1
        data = [1] * row.shape[0]
        ind = max(max(col), max(row)) + 1
        OriginalNetwork = coo_matrix((data, (row, col)), shape=(ind, ind)).toarray()
        OriginalNetwork += OriginalNetwork.T
        Odiag = np.diagonal(OriginalNetwork)
        if max(Odiag) != 0:
            OriginalNetwork -= np.diag(Odiag)
        return OriginalNetwork
    if choose == "4":
        arr = np.loadtxt('dataset/world_trade.txt', dtype='int')
        row = arr[:, 0] - 1
        col = arr[:, 1] - 1
        data = [1] * row.shape[0]
        ind = max(col) + 1
        OriginalNetwork = coo_matrix((data, (row, col)), shape=(ind, ind)).toarray()
        OriginalNetwork += OriginalNetwork.T
        Odiag = np.diagonal(OriginalNetwork)
        if max(Odiag) != 0:
            OriginalNetwork -= np.diag(Odiag)
        return OriginalNetwork
    if choose == "5":
        arr = np.loadtxt('dataset/USAir.txt', dtype='int')
        data = arr[:, 2]
        row = arr[:, 0] - 1
        col = arr[:, 1] - 1
        ind = max(col) + 1
        data = 1 / (1 + np.exp(-data))
        OriginalNetwork = coo_matrix((data, (row, col)), shape=(ind, ind)).toarray()
        OriginalNetwork += OriginalNetwork.T
        Odiag = np.diagonal(OriginalNetwork)
        if max(Odiag) != 0:
            OriginalNetwork -= np.diag(Odiag)
        return OriginalNetwork
    if choose == "6":
        arr = np.loadtxt('dataset/Celegans_w.txt', dtype='int')
        data = arr[:, 2]
        row = arr[:, 0] - 1
        col = arr[:, 1] - 1
        ind = max(col) + 1
        OriginalNetwork = coo_matrix((data, (row, col)), shape=(ind, ind)).toarray()
        OriginalNetwork += OriginalNetwork.T
        Odiag = np.diagonal(OriginalNetwork)
        if max(Odiag) != 0:
            OriginalNetwork -= np.diag(Odiag)
        return OriginalNetwork

