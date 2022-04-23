import gc
import os
import scipy
import torch
import random
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import trange

from sklearn.model_selection import train_test_split

from model import GCN
from utils import rocks_cache,cache_with_callable,print_graph_detail, EarlyStopping, accuracy, macro_f1

warnings.filterwarnings("ignore")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class TextGCNTrainer:
    def __init__(self,
                 model,
                 seed,
                 lr,
                 device,
                 dataset,
                 max_epoch,
                 early_stopping,

                 graph_nodes,  # 图的节点数，也是 feature 的 维度
                 nhid,  # hidden layer shape
                 dropout,  # dropout

                 adjacency_matrix,  # 图的邻接矩阵 adjacency_matrix
                 features,  # 特征矩阵features
                 target,  # target 向量
                 nclass,  # 类别数
                 train_lst,
                 valid_lst,
                 test_lst,
                 ):
        # 训练参数初始化
        self.lr = lr
        self.device = device
        self.dataset = dataset
        self.max_epoch = max_epoch
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.earlystopping = EarlyStopping(early_stopping)

        # 模型参数初始化
        self.nhid = nhid
        self.model = model
        self.nclass = nclass
        self.dropout = dropout
        self.graph_nodes = graph_nodes
        print(self.__dict__)
        # 数据初始化
        self.adjacency_matrix = adjacency_matrix
        self.features = features
        self.target = target
        self.train_lst = train_lst
        self.valid_lst = valid_lst
        self.test_lst = test_lst

    def move(self):
        # data move
        self.adjacency_matrix = self.adjacency_matrix.to(self.device)
        self.features = self.features.to(self.device)
        self.target = torch.tensor(self.target).long().to(self.device)
        self.train_lst = torch.tensor(self.train_lst).long().to(self.device)
        self.valid_lst = torch.tensor(self.valid_lst).long().to(self.device)
        self.test_lst = torch.tensor(self.test_lst).long().to(self.device)

    def fit(self):
        """
        初始化训练，设置优化器和损失函数，最后进行训练
        """
        self.model = self.model(nfeat=self.graph_nodes,
                                nhid=self.nhid,
                                nclass=self.nclass,
                                dropout=self.dropout)
        self.model = self.model.to(self.device)
        print(f"==> Start Train ... \n {self.model.parameters}")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model_param = sum(param.numel()
                               for param in self.model.parameters())
        self.move()
        self.train()

    def train(self):
        """
        训练
        """
        with  trange(self.max_epoch) as epoch_pbar:
            for epoch in epoch_pbar:
                self.model.train()
                self.optimizer.zero_grad()

                # forward
                logits = self.model.forward(self.features, self.adjacency_matrix)
                loss = self.criterion(logits[self.train_lst], self.target[self.train_lst])

                # backward
                loss.backward()
                self.optimizer.step()

                # valid data
                valid_desc = self.valid(self.valid_lst)

                # set description
                desc = dict(**{"epoch": epoch, "train_loss": loss.item()}, **valid_desc)
                epoch_pbar.set_description(self.description_to_string(desc))
                # check loss
                if self.earlystopping(valid_desc["valid_loss"]):
                    raise Exception("earlystopping error")
                    break

    @torch.no_grad()
    def valid(self, x, prefix='valid'):
        self.model.eval()
        with torch.no_grad():
            logits = self.model.forward(self.features, self.adjacency_matrix)
            loss = self.criterion(logits[x],
                                  self.target[x])
            acc = accuracy(logits[x],
                           self.target[x])
            f1, precision, recall = macro_f1(logits[x],
                                             self.target[x],
                                             num_classes=self.nclass)

            desc = {
                f"{prefix}_loss": loss.item(),
                "acc": acc,
                "macro_f1": f1,
                "precision": precision,
                "recall": recall,
            }
        return desc

    @classmethod
    def description_to_string(cls, desc):
        string = ""
        for key, value in desc.items():
            if isinstance(value, int):
                string += f"{key}:{value} "
            else:
                string += f"{key}:{value:.4f} "
        return string

    @torch.no_grad()
    def test(self):
        test_desc = self.valid(self.test_lst, prefix="test")
        test_desc["model_param"] = self.model_param
        print(test_desc)
        return test_desc


class TrainProcess:
    def __init__(self,
                 dataset,
                 times,
                 model,
                 nhid=200,
                 max_epoch=200,
                 dropout=0.5,
                 val_ratio=0.1,
                 early_stopping=10,
                 lr=0.02,
                 path=None,
                 ):

        self.dataset = dataset  # 数据集名称
        self.times = times  # 训练次数
        self.model = model  # 模型

        if path is None:
            self.path = os.path.dirname(os.path.abspath(__file__))  # 项目根目录
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')  # 设备
        self.nhid = nhid  # hidden layer shape
        self.max_epoch = max_epoch  # 最大epoch
        self.dropout = dropout
        self.val_ratio = val_ratio
        self.early_stopping = early_stopping
        self.lr = lr

        # 读取图，并转为 张量
        self.graph = cache_with_callable(f'{self.dataset}_graph',self.read_graph) # self.graph
        print_graph_detail(self.graph)
        self.adjacency_matrix = cache_with_callable(f'{self.dataset}_adjacency_matrix',self.graph_to_tensor) # self.adjacency_matrix
        # 得到 特征矩阵
        self.features = cache_with_callable(f'{self.dataset}_features',self.get_features) # self.features
        # 读取语料的描述文件(或者是 语料的 label)
        self.get_target()  # self.target ,self.nclass

    def train(self):
        seed_lst = []
        for i, seed in enumerate(random.sample(range(0, 100000), self.times)):
            print(f"\n\n==>Training {i}, seed:{seed}  ...")
            # 设置和保存随机种子
            self.seed = seed
            seed_lst.append(self.seed)

            # 这个没有放在初始化部分，是因为该部分需要使用随机种子
            # 分割 测试数据集和 训练数据集
            # https://towardsdatascience.com/how-to-split-data-into-three-sets-train-validation-and-test-and-why-e50d22d3e54c

            self.train_valid_test_split()  # self.train_lst ,self.valid_lst, self.test_lst

            trainer = TextGCNTrainer(seed=self.seed,
                                     lr=self.lr,
                                     device=self.device,
                                     dataset=self.dataset,
                                     max_epoch=self.max_epoch,
                                     early_stopping=self.early_stopping,

                                     nhid=self.nhid,
                                     model=self.model,
                                     nclass=self.nclass,
                                     dropout=self.dropout,
                                     graph_nodes=self.graph.number_of_nodes(),

                                     adjacency_matrix=self.adjacency_matrix,
                                     features=self.features,
                                     target=self.target,
                                     train_lst=self.train_lst,
                                     valid_lst=self.valid_lst,
                                     test_lst=self.test_lst,

                                     )
            trainer.fit()
            trainer.test()
            del trainer
            # 垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print("Training Success")

    def train_valid_test_split(self):
        """
        train_valid_test_split
        """
        train_data = list()
        test_data = list()
        target_data_path = self.path+f"/datasets/describe/{self.dataset}.txt"
        with open(target_data_path, mode='r') as f:
            for indx, item in enumerate(f):
                # 20ng 已经划分了 测试数据和 训练数据
                if len(item.split("\t")) >= 2 and item.split("\t")[1] in {"train", "training", "20news-bydate-train"}:
                    train_data.append(indx)
                else:
                    test_data.append(indx)
        if train_data:
            self.train_lst, self.valid_lst = train_test_split(train_data,
                                                            test_size=self.val_ratio,
                                                            shuffle=True,
                                                            random_state=self.seed)
        else:
            train_data, self.test_lst = train_test_split(test_data,
                                                test_size=self.val_ratio,
                                                shuffle=True,
                                                random_state=self.seed)
            self.train_lst, self.valid_lst = train_test_split(train_data,
                                                            test_size=self.val_ratio,
                                                            shuffle=True,
                                                            random_state=self.seed)

    def get_target(self):
        """获取描述文件中的label 并 构造 target 向量
        """
        target_data_path = self.path+f"/datasets/describe/{self.dataset}.txt"
        data = np.array(pd.read_csv(target_data_path, sep="\t", header=None))
        target = np.array(pd.read_csv(target_data_path, sep="\t", header=None)[len(data[0])-1])

        label_idx_mapping = {label: indx for indx,
                             label in enumerate(set(target))}
        self.target = [label_idx_mapping[label]
                       for label in target]  # target 向量
        self.nclass = len(label_idx_mapping)  # 类别

    def get_features(self):
        """获取特征矩阵
        """
        graph_nodes = self.graph.number_of_nodes()
        row = list(range(graph_nodes))
        col = list(range(graph_nodes))
        value = [1.] * graph_nodes
        shape = (graph_nodes, graph_nodes)
        indices = torch.from_numpy(
            np.vstack((row, col)).astype(np.int64))
        values = torch.FloatTensor(value)
        shape = torch.Size(shape)
        self.features = torch.sparse.FloatTensor(indices, values, shape)
        return self.features

    def read_graph(self):
        graph_data_path = self.path+f"/temp/graph/{self.dataset}.txt"
        print(f"==> waiting for read dataset graph_data:{self.dataset}, graph_data_path: {graph_data_path} <==")
        self.graph = nx.read_weighted_edgelist(graph_data_path, nodetype=int)
        return self.graph

    def graph_to_tensor(self):
        """
        networkx.Graph 转为 邻接矩阵 再转为 torch tensor
        """

        """
        >>> G.add_edge(0, 1, weight=2)
        0
        >>> G.add_edge(1, 0)
        0
        >>> G.add_edge(2, 2, weight=3)
        0
        >>> G.add_edge(2, 2)
        1
        >>> S = nx.to_scipy_sparse_matrix(G, nodelist=[0, 1, 2])
        >>> print(S.todense())
        [[0 2 0]
        [1 0 0]
        [0 0 4]]

        """

        print(f"==> waiting for graph_data to scipy_sparse_matrix to torch_sparse_tensor <==")
        sparse_matrix = nx.to_scipy_sparse_matrix(self.graph,
                                                  nodelist=list(
                                                      range(self.graph.number_of_nodes())),
                                                  weight='weight',
                                                  dtype=np.float)
        need_normalize_adjacency_matrix = sparse_matrix + sparse_matrix.T.multiply(
            sparse_matrix.T > sparse_matrix) - sparse_matrix.multiply(sparse_matrix.T > sparse_matrix)
        self.adjacency_matrix = self.sparse_mx_to_torch_sparse_tensor(
            self.normalize_adjacency_matrix(need_normalize_adjacency_matrix))
        return self.adjacency_matrix

    def normalize_adjacency_matrix(self, adjacency_matrix):
        """正规化邻接矩阵"""
        adjacency_matrix = scipy.sparse.coo_matrix(adjacency_matrix)
        rowsum = np.array(adjacency_matrix.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = scipy.sparse.diags(d_inv_sqrt)
        return adjacency_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """
        将一个scipy稀疏矩阵转换为torch稀疏张量
        :sparse_mx scipy  sparse matrix 转为 torch 张量
        """
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == '__main__':
    import utils
    utils.CACHE_SWITCH = True
    TrainProcess("20ng", 1, GCN,max_epoch=200).train()
