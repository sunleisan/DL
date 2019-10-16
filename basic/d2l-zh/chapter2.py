from mxnet import nd, autograd
import numpy as np

# 创建
x = nd.arange(12)
x.reshape(3, -1)  # -1自行推断
# 创建张量
y = nd.zeros((2, 3, 4))
print(y)
r = nd.random.normal(0, 1, shape=(3, 4))

# 不同形状的元素相加导致的广播机制

# 2.3 自动求梯度
print("2.3 自动求梯度")
# x=[0,1,2,3]^T 特定点的梯度
x = nd.arange(4).reshape((4, 1))
# 申请存储梯度的内存,默认x.grand=[0,0,0,0],调用y.backward()后可以求得梯度[0,4,8,12]
x.attach_grad()
# with 上下文管理协议,用于对资源进行访问,能够自动管不文件，自动释放资源
with autograd.record():
    # y=2x1^2+2x2^2+2x3^2+2x4^2
    y = 2 * nd.dot(x.T, x)  # 定义求导的函数
y.backward()  # 计算梯度
print(x.grad)