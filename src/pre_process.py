import os
import re
import torch
from collections import Counter
from collections import defaultdict
import numpy as np
import pandas as pd
import time
import networkx as nx
import datetime
from tqdm import tqdm
from corpus_process import StringProcess
from sklearn.metrics.pairwise import cosine_similarity
from utils import cache_with_callable


class PreProcess:
    """
    前处理
    """

    def __init__(self, dataset, encoding=None, min_freq=5):
        path = os.path.dirname(os.path.abspath(__file__))  # 项目根目录
        self.min_freq = min_freq  # 最小词频

        # 原始数据地址
        origin_path = path+"/datasets/origin"
        # 图数据地址
        graph_path = path+"/temp/graph"

        corpus_path = path+"/datasets/corpus"
        if not os.path.exists(corpus_path):
            os.makedirs(corpus_path)
        target_path = path+"/datasets/target"
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        knn_path = path+"/temp/knn"
        if not os.path.exists(knn_path):
            os.makedirs(knn_path)
        features_path = path+"/temp/features"
        if not os.path.exists(features_path):
            os.makedirs(features_path)

        self.dataset = dataset
        self.origin_data_path = f"{origin_path}/{dataset}.csv"
        # 语料生成结果存储地址
        self.corpus_path_path = f"{corpus_path}/{dataset}.txt"
        self.target_path_path = f"{target_path}/{dataset}.txt"
        # graph
        self.graph_data_path = f"{graph_path}/{dataset}.txt"
        # graph
        self.knn_data_path = f"{knn_path}/{dataset}.txt"
        # features
        self.features_data_path = f"{features_path}/{dataset}.txt"

        # 字典
        self.context_dct = defaultdict(dict)

        # 编码
        self.encoding = encoding

    def build_features_corpus(self, length=None):
        datas = np.array(pd.read_csv(
            self.origin_data_path, sep=",", header=None))
        textual_data_list = []
        catalog_data_list = []
        target = []
        sp = StringProcess()
        count = 0
        for i in tqdm(datas[1:], desc="build_features_corpus ..."):
            log_time = datetime.datetime.strptime(i[1], "%Y/%m/%d %H:%M")
            textual_data_list.append(
                {
                    "url": " URL".join(sp.clean_str(i[3]).split(' ')) if sp.clean_str(i[3]) else 'urlEmpty',
                    "refer": " REFER".join(sp.clean_str(i[7]).split(' ')) if sp.clean_str(i[7]) else "referEmpty",
                    "ua": " UA".join(sp.clean_str(i[8]).split(' ')) if sp.clean_str(i[8]) else "uaEmpty"
                }
            )
            catalog_data_list.append(
                {
                    "content_type": "contentType"+sp.clean_str(i[3]).split(' ')[-1] if sp.clean_str(i[3]) else "contentTypeEmpty",
                    "night": "timeNight" if log_time.hour < 5 else "timeDaylight",
                    "host": "HOST"+i[0],
                    "method": "method"+i[2],
                    "protocol": "protocol"+i[4],
                    "rv": "rvCode"+i[5],
                    "have_refer": "haveRefer" if i[7] and i[7] != "-" else "notHaveRefer",
                    "length": "lengthEmpty" if int(i[6]) == 0 else "lengthShort" if int(i[6]) < 300 else "lengthLong",
                }
            )
            target.append(i[9])
            count += 1
            if length is not None and count >= length:
                break
        with open(self.corpus_path_path, mode='w') as fout:
            for t, c in zip(textual_data_list, catalog_data_list):
                lines_str = ' '.join(c.values())
                lines_str += ' '.join(t.values())
                fout.write(lines_str)  # 写入到 语料词序列文件中
                fout.write(" \n")
        with open(self.target_path_path, mode='w') as fout:
            for i in target:
                fout.write(f"{i}\n")

    def build_features_matrix(self):
        """获取特征矩阵
        """
        total_document = 0
        with open(self.target_path_path, mode='r') as data:
            total_document = len(data.readlines())

        graph = nx.read_weighted_edgelist(self.graph_data_path, nodetype=int)
        graph_nodes = graph.number_of_nodes()

        features = []
        for d_id in tqdm(range(total_document), desc='get_features_tensor...'):
            values = []
            for w_id in range(graph_nodes-total_document, graph_nodes):
                values.append(1 if graph.get_edge_data(d_id, w_id) else 0)
            features.append(values)
        features = torch.from_numpy(np.array(features, dtype=int))

        np.savetxt(self.features_data_path, features, fmt='%d')
        return features

    def build_knn_graph(self, k):
        print(
            f"==> waiting for build knn graph data:{self.dataset}, knn_graph_data_path: {self.knn_data_path} <==")
        features = np.genfromtxt(self.features_data_path, dtype=np.int16)
        dist = cosine_similarity(features)
        inds = []
        for i in tqdm(range(dist.shape[0]), desc="writing knn argpartition"):
            ind = np.argpartition(dist[i, :], -(k + 1))[-(k + 1):]
            inds.append(ind)
        with open(self.knn_data_path, 'w') as f:
            for i, v in tqdm(enumerate(inds), desc="writing knn data"):
                for vv in v:
                    if int(i) < int(vv):
                        f.write('{} {}\n'.format(i, vv))


if __name__ == "__main__":
    PreProcess("logs").build_features_corpus()  # 构建语料
    # PreProcess("logs").build_features_matrix()  # 根据图构建特征
    # PreProcess("logs").build_knn_graph(5)  # 构建图结构
