from mxnet import nd
import numpy as np

# 创建
x = nd.arange(12)
x.reshape(3, -1)  # -1自行推断
# 创建张量
y = nd.zeros((2, 3, 4))
r = nd.random.normal(0, 1, shape=(3, 4))

# 不同形状的元素相加导致的广播机制
