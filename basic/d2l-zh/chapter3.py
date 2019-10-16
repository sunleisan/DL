from mxnet import nd
from time import time
from matplotlib import pyplot as plt
from IPython import display
import random


def init_figure(size=(3.5, 2.5)):
    display.set_matplotlib_formats('svg')  # 矢量图
    plt.rcParams['figure.figsize'] = size


# 每次返回数据集中随机的部分数据
def data_iter(data_sets, labels, batch_size=10):
    num_examples = len(data_sets)  # 数据长度
    indices = list(range(num_examples))  # 数据读取次序
    # random.shuffle 将序列中的元素随机排序
    random.shuffle(indices)  # 打乱读取次序
    # range函数(a,b,c)从a开始不大于b的数,步长为c
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i:min(i + batch_size, num_examples)])
        # 用于迭代，返回数据,提供索引数组，返回对应值的数组
        yield data_sets.take(j), labels.take(j)


def linear_node():
    # 使用矢量计算表达式
    a = nd.ones(1000)  # 1000维的向量
    b = nd.ones(1000)
    start = time()
    c = nd.zeros(1000)
    for i in range(1000):
        c[i] = a[i] + b[i]
    print(time() - start)
    start = time()
    d = a + b
    print(time() - start)
    # 线性回归模型# 1 生成数据集
    feature = 2  # 特征值个数
    data_set = 1000  # 数据集个数
    true_w = [2, -3.4]  # 模型真实权重 二维向量
    true_b = 4.2  # 真实偏移量
    # 正态分布,scale:标准差,loc:均值,shape:输出数组形状
    X = nd.random.normal(scale=1, loc=0, shape=(data_set, feature))
    # 计算模型输出值
    labels = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
    # 加上噪音参数
    labels += nd.random.normal(scale=0.01, shape=labels.shape)
    init_figure()
    plt.scatter(X[:, 1].asnumpy(), labels.asnumpy(), 1)
    plt.show()
