#!/usr/bin/env python
import math
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907

    参考 https://github.com/tkipf/pygcn
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 由于weight是可以训练的，因此使用parameter定义
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # 由于bias是可以训练的，因此使用parameter定义
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else :
            self.register_parameter('bias', None)
        self.reset_parameter()

    def reset_parameter(self):
        # size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数  size（1）为行
        stdv = 1. / math.sqrt(self.weight.size(1))
        # uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input, adj) :
        # torch.mm(a, b)是矩阵a和b矩阵相乘，torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等
        # torch.spmm(a,b)是稀疏矩阵相乘
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else :
            return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(Module):
    """
    两层的gcn
    
    参考：https://mp.weixin.qq.com/s/CpDZEqo14X_lCBh6i7feIA
    一个两层的GCN可以允许信息在最多两阶节点之间传递。因此，尽管图中文档和文档没有直接的边，但是两层的GCN允许文档对之间的信息交换。在我们的初步实验中。我们发现，双层GCN的性能优于单层GCN，而多层GCN的性能并没有提高。这类似于(Kipf and Welling 2017) 和 (Li, Han, and Wu 2018)的结果。

    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = torch.relu(x)
        x = torch.dropout(x, self.dropout, train=self.training)
        x = self.gc2(x, adj)
        return x

