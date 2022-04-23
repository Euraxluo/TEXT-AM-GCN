import os
from collections import Counter

import networkx as nx

import itertools
import math
from collections import defaultdict
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from utils import print_graph_detail


def get_window(content_lst, window_size):
    """
    找出窗口
    :param content_lst: 文档列表
    :param window_size: 窗口大小
    :return: w(i) , w(i, j) , 窗口数
    """
    word_window_freq = defaultdict(int)  # w(i)  单词在窗口单位内出现的次数
    word_pair_count = defaultdict(int)  # w(i, j)
    windows_len = 0  # 窗口数
    for document in tqdm(content_lst, desc="build window"):
        windows = list()

        if isinstance(document, str):
            words = document.split()
        else:
            raise TypeError("document must str")

        length = len(words)

        # 如果小于 窗口大小，将词序列直接append，否则构建滑动窗口 词序列列表
        if length <= window_size:
            windows.append(words)
        else:
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(list(set(window)))

        for window in windows:
            for word in window:
                word_window_freq[word] += 1

            for word_pair in itertools.combinations(window, 2):
                word_pair_count[word_pair] += 1

        windows_len += len(windows)
    return word_window_freq, word_pair_count, windows_len


def cal_pmi(W_ij, W, word_freq_1, word_freq_2):
    """
    pmi(i,j) = log( (w(i,j)/w_len) / (w(i)/w_len)*(w(j)/w_len))
    """
    p_i = word_freq_1 / W
    p_j = word_freq_2 / W
    p_i_j = W_ij / W
    pmi = math.log(p_i_j / (p_i * p_j))

    return pmi


def get_word_pmi_list(windows_len, word_pair_count, word_window_freq, threshold):
    """
    windows_len:窗口数
    word_pair_count: w(i, j)
    word_window_freq: w(i)
    threshold: 阈值
    :return list of pmi(i,j)
    """
    word_pmi_lst = list()  # list of pmi(i,j)
    for word_pair, W_i_j in tqdm(word_pair_count.items(), desc="Calculate pmi between words"):
        # word_pair ('cb', 'dean') (word1,word2)
        # W_i_j ('cb', 'dean') count of all document windwos
        # word_window_freq :count of all document windwos word
        word_freq_1 = word_window_freq[word_pair[0]]  # w(i)
        word_freq_2 = word_window_freq[word_pair[1]]  # w(j)

        pmi = cal_pmi(W_i_j, windows_len, word_freq_1, word_freq_2)
        # pmi小于阈值的 进行忽略
        if pmi <= threshold:
            continue
        word_pmi_lst.append([word_pair[0], word_pair[1], pmi])
    return word_pmi_lst


def get_pmi_edge(content_lst, window_size=20, threshold=0.):
    """
    window_size:文档内，滑动窗口计算大小
    threshold:pmi(i,j) 小于阈值的会被舍弃，即不会建立 word-word edge
    :return list of pmi(i,j)  形如:List[List[word1:str,word2:str,pmi(word1,word2)[>threshold]:float]]
    """
    if isinstance(content_lst, str):
        content_lst = list(open(content_lst, "r"))
    print("pmi read file len:", len(content_lst))

    # 获取 window，w(i) , w(i, j) , 窗口数
    word_window_freq, word_pair_count, windows_len = get_window(
        content_lst, window_size=window_size)
    # 计算 pmi(i,j) 并返回序列
    word_pmi_list = get_word_pmi_list(
        windows_len, word_pair_count, word_window_freq, threshold)

    print("Total number of edges between word:", len(word_pmi_list))
    return word_pmi_list


class GraphBuildProcess:
    def __init__(self, dataset):
        path = os.path.dirname(os.path.abspath(__file__))  # 项目根目录
        # 处理后的语料地址
        clean_corpus_path = path+"/temp/clean_corpus"
        # 图构建结果存储地址
        self.graph_path = path+"/temp/graph"
        # 数据集名称
        self.dataset = dataset
        if not os.path.exists(self.graph_path):
            os.makedirs(self.graph_path)

        # 单词映射
        self.word2id = dict()
        # networkx 对象
        self.g = nx.Graph()
        self.content = f"{clean_corpus_path}/{dataset}.txt"
        print(f"\n==> 现在的数据集是:{dataset},地址: {self.content} <==")
        # **文档节点**之间的边的重量和一个单词节点是文档中这个词的词频-逆文档频率(TF-IDF)
        self.get_document_word_tfidf_edge()
        # 点互信息(PMI)，一种测量单词关联性的流行方法，来计算**两个单词节点**之间的权重
        self.get_word_word_pmi_edge()
        # 保存数据
        print("waiting fot write weighted_edgelist")
        nx.write_weighted_edgelist(self.g, f"{self.graph_path}/{self.dataset}.txt")
        print("GraphBuild success")

    def get_word_word_pmi_edge(self):
        """
        build word-word edges with pmi(word1,word2)
        """
        start = time.perf_counter()
        pmi_edge_lst = get_pmi_edge(
            self.content, window_size=20, threshold=0.0)

        for word1, word2, pmi in pmi_edge_lst:
            word_indx1 = self.node_num + self.word2id[word1]
            word_indx2 = self.node_num + self.word2id[word2]
            if word_indx1 == word_indx2:
                continue
            self.g.add_edge(word_indx1, word_indx2, weight=pmi)

        print("get word-word edges build time", time.perf_counter() - start)
        print_graph_detail(self.g)

    def get_document_word_tfidf_edge(self):
        """
        build document-word edges with tfidf
        """
        # 获得tfidf权重矩阵（sparse）和单词列表
        # 构建了self.node_num ，self.vocab_lst ，self.word2id
        # node 是 document node ，vocab_lst 是词表，word2id 是词表索引，tfidf_vec 是为文档的tfidf 矩阵
        start = time.perf_counter()
        tfidf_vec = self.get_tfidf_vec()

        count_lst = list()  # 统计每个句子的长度
        for ind, row in tqdm(enumerate(tfidf_vec),
                             desc="build document-word edges with tfidf"):
            count = 0
            for col_ind, value in zip(row.indices, row.data):
                # 对应 词典 索引为 col_ind 的词，对应的tfidf为value
                # row.todense().tolist() 能够获取到一个稀疏矩阵
                word_ind = self.node_num + col_ind  # 单词索引，总文档数偏移 col_index
                # ind 是 文档索引,word_ind 是单词索引，因此该边为 文档中单词共现(document-word edges)
                self.g.add_edge(ind, word_ind, weight=value)
                count += 1
            count_lst.append(count)  # (document-word edges)个数

        print("document-word edges build time:", time.perf_counter() - start)
        print_graph_detail(self.g)

    def get_tfidf_vec(self):
        """
        https://www.cnblogs.com/171207xiaohutu/p/10083545.html

        https://blog.csdn.net/pnnngchg/article/details/85054243

        https://blog.csdn.net/Elenstone/article/details/105134863

        学习获得tfidf矩阵，及其对应的单词序列，然后生成词表
        :param content_lst:
        :return:
        """
        text_tfidf = Pipeline([
            ("vect", CountVectorizer(min_df=1,
                                     max_df=1.0,
                                     token_pattern=r"\S+",
                                     )),  # BOW 构建词表
            ("tfidf", TfidfTransformer(norm=None,
                                       use_idf=True,
                                       smooth_idf=False,
                                       sublinear_tf=False
                                       ))  # TF-IDF 矩阵
        ])

        tfidf_vec = text_tfidf.fit_transform(open(self.content, "r"))  # tfidf
        # (18846,42757) (document count,world count)
        print("tfidf_vec shape:", tfidf_vec.shape)
        print("tfidf_vec type:", type(tfidf_vec))

        self.node_num = tfidf_vec.shape[0]  # node is document node

        # 映射单词，建立词表
        self.vocab_lst = text_tfidf["vect"].get_feature_names()  # 获取词表
        print("vocab_lst len:", len(self.vocab_lst))
        for ind, word in enumerate(self.vocab_lst):  # 构建反向索引，得到词表索引
            self.word2id[word] = ind

        return tfidf_vec


if __name__ == '__main__':
    GraphBuildProcess("20ng")
