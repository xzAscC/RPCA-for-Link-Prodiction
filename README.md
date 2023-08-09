# RPCA-for-Link-Prodiction

对Link Predicton via matrix completion的复现

数据集存放在dataset中
通过运行Python文件main.py可以运行该代码

运行时需要输入一个变量：选定的数据集，输入1-6之间的整数，选定需要运行的数据集
运行次数loopTimes预先定10；训练集预先定为整个网络的0.90，保存在参数ratio中；

参数lmbda在代码中预先固定，可以通过调整不同的lmbda调整低秩与稀疏比例
但是需要注意lmbda如果设置的太小或者太大，预测都将表现不佳
