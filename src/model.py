#!/usr/bin/env python
import math
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

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
        # x = torch.sigmoid(x) # 如果直接使用 可以考虑稍微魔改一下模型
        return x



class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class SFGCN(nn.Module):
    """
    https://cloud.tencent.com/developer/article/1688778

    """
    def __init__(self, nfeat, nclass, nhid1, nhid2, dropout):
        super(SFGCN, self).__init__()

        self.SGCN1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN2 = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(nhid2)
        self.tanh = nn.Tanh()

        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, sadj, fadj):
        """
        为了充分铺捉到特征空间中的结构信息，
        我们使用由节点特征生成的k近邻图作为特征结构图 (Feature Graph)。
        通过同时利用特征图 (Feature Graph) 和拓扑图 (Topology Graph)，
        我们将节点特征传播到拓扑空间和特征空间，
        并利用两个特殊的图卷积模块 (Specific Convolution) 在这两个空间中提取两个特殊的嵌入。
        考虑到两个空间之间有存在共同特性的可能，
        我们设计了一个具有参数共享策略的公共图卷积模块 (Common Convolution) 来提取它们共有的嵌入。
        然后我们进一步利用注意机制 (Attention Mechanism) 来自动学习不同嵌入的重要性权重，
        从而自适应地融合它们。通过这种方式，节点标签能够监督学习过程，
        自适应调整权重，提取出图数据中与任务相关性最强的信息。
        此外，我们设计了一致性约束 (Consistency Constraint) 
        和差异性约束 (Disparity Constraint)，
        以确保所学习的节点表示的一致性和差异
        :param x: _description_
        :type x: _type_
        :param sadj: _description_
        :type sadj: _type_
        :param fadj: _description_
        :type fadj: _type_
        :return: _description_
        :rtype: _type_
        """
        emb1 = self.SGCN1(x, sadj) # Special_GCN out1 -- sadj structure graph
        com1 = self.CGCN(x, sadj)  # Common_GCN out1 -- sadj structure graph
        com2 = self.CGCN(x, fadj)  # Common_GCN out2 -- fadj feature graph
        emb2 = self.SGCN2(x, fadj) # Special_GCN out2 -- fadj feature graph
        Xcom = (com1 + com2) / 2
        ##attention
        emb = torch.stack([emb1, emb2, Xcom], dim=1)
        # emb = torch.stack([emb1, com1], dim=1)
        emb, att = self.attention(emb)
        output = self.MLP(emb)
        # return output#, att, emb1, com1, com2, emb2, emb
        return output, att, emb1, com1, com2, emb2, emb
