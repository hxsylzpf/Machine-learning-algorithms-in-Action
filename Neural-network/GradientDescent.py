import numpy as np
import random


class NetWork(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


net = NetWork([2, 3, 1])
print net.num_layers
print net.sizes
print net.biases
print net.weights


# np.random.rand(y, 1): 随机从正态分布(均值0, 方差1)中生成
# net.weights[1] 存储连接第二层和第三层的权重 (Python索引从0开始数)

def feedForward(self, a):
    """Return the output of the network if "a" is input."""
    for b, w in zip(self.biases, self.weights):
        a = sigmoid(np.dot(w, a) + b)
    return a
