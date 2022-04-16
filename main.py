from rpca import *
from load_data import *

if __name__ == "__main__":
    choose = input("输入选择的数据集：1.Jazz 2.Contact 3. Political 4. world_trade 5. UsAir 6. elegans\n")
    G = loadData(choose)
    loopTimes = 10
    precision = []
    lmbda = [0.13, 0.12, 0.07, 0.12, 0.10, 0.10]
    ratio = 0.9
    for i in range(loopTimes):
        train, test = divideNetwork(G, ratio)
        X, E = alm_rpca(train, lmbda[int(choose) - 1])
        X = X + X.T
        # if (test == np.zeros((Networks.shape[0], Networks.shape[1]))).all():
        # if (Networks == train).all():
        # if((test + train == G).all()):
        #     print(1)
        p = compute_precision(X, train, test)
        precision.append(p)
    # print(precision)
    print(np.mean(precision))
    # print(np.std(precision))
