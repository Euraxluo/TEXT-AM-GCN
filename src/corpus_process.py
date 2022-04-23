import os
import re
from collections import Counter
from collections import defaultdict
import numpy as np

from tqdm import tqdm


class StringProcess:
    def __init__(self):
        self.other_char = re.compile(r"[^A-Za-z0-9(),!?\'\`]", flags=0)
        self.num = re.compile(r"[+-]?\d+\.?\d*", flags=0)
        self.url = re.compile(
            r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", flags=0)
        self.stop_words = None
        self.nlp = None

    def clean_str(self, string):
        """
        主要是为英文分词做准备 准备tokenize 进行字符串清理
        参考 https://www.ylkz.life/deeplearning/p10550146/
        """
        string = re.sub(self.other_char, " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip().lower()

    def norm_str(self, string):
        string = re.sub(self.other_char, " ", string)

        if self.nlp is None:
            from spacy.lang.en import English
            self.nlp = English()

        new_doc = list()
        doc = self.nlp(string)
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if token.is_digit:
                token = "[num]"
            else:
                token = token.text

            new_doc.append(token)

        return " ".join(new_doc).lower()

    def lean_str_sst(self, string):
        """
        Tokenization/string cleaning for the SST yelp_dataset
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(self.other_char, " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def remove_stopword(self, string):
        """
        将那些英文停用词跳过
        """
        if self.stop_words is None:
            """
            英文停用词
            """
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))

        if type(string) is str:
            string = string.split()

        new_string = list()
        for word in string:
            if word in self.stop_words:
                continue
            new_string.append(word)

        return " ".join(new_string)

    def replace_num(self, string):
        result = re.sub(self.num, '<num>', string)
        return result

    def replace_urls(self, string):
        result = re.sub(self.url, '<url>', string)
        result = ' '.join(re.split(' +|\n+', result)).strip()
        return result


class CorpusProcess:
    """
    我们首先通过清理和标记文本(Kim 2014)对所有数据集进行预处理。
    然后，我们删除了NLTK定义的停用词和20NG、R8、R52和Ohsumed中出现少于5次的低频字。
    唯一的例外是MR，我们在清理和标记原始文本后没有删除单词，因为文档非常短。(MR文本长度为)
    表1总结了预处理数据集的统计数据

    """

    def __init__(self, dataset, encoding=None, min_freq=5):
        path = os.path.dirname(os.path.abspath(__file__)) # 项目根目录
        self.min_freq = min_freq  # 最小词频

        # 语料地址
        corpus_path = path+"/datasets/corpus"
        # 处理结果存储地址
        clean_corpus_path = path+"/temp/clean_corpus"
        if not os.path.exists(clean_corpus_path):
            os.makedirs(clean_corpus_path)

        self.dataset = dataset
        self.corpus_name = f"{corpus_path}/{dataset}.txt"
        self.save_name = f"{clean_corpus_path}/{dataset}.txt"
        print(self.dataset)
        print(self.corpus_name)
        print(self.save_name)
        # 字典
        self.context_dct = defaultdict(dict)

        # 编码
        self.encoding = encoding

        self.clean_text()
        print("CorpusProcess success")

    def remove_less_word(self, lines_str, word_st):
        return " ".join([word for word in lines_str.split() if word in word_st])

    def clean_text(self):
        """
        处理语料
        """
        sp = StringProcess()

        # 整个语料数据集的分词序列
        word_lst = list()

        # 打开语料
        with open(self.corpus_name, mode="rb", encoding=self.encoding) as fin:
            for indx, item in tqdm(enumerate(fin), desc="clean the text"):
                data = item.strip().decode('latin1')  # 拉丁文字解码
                data = sp.clean_str(data)  # 清理字符串
                data = sp.remove_stopword(data)  # 删除英文停用词
                word_lst.extend(data.split())

        # 整个语料数据集，建立词频表，去掉小于最小词频的词
        word_st = set()
        for word, value in Counter(word_lst).items():
            if value < self.min_freq:
                continue
            word_st.add(word)

        # 再对语料数据的每一行进行处理
        doc_len_lst = list()
        with open(self.save_name, mode='w') as fout:
            with open(self.corpus_name, mode="rb", encoding=self.encoding) as fin:
                for line in tqdm(fin, desc="build lines word str seq"):
                    lines_str = line.strip().decode('latin1')
                    lines_str = sp.clean_str(lines_str)
                    if len(lines_str) < 300:  # 原 论文 是直接不处理 mr 数据，我这里改成 不处理 长度小于 300 以下的文本
                        lines_str = sp.remove_stopword(
                            lines_str)  # 字符串清理后的英文语料
                        lines_str = self.remove_less_word(
                            lines_str, word_st)  # 根据词频表建立词序列，这些词序列去掉了最小词频以下的词

                    fout.write(lines_str)  # 写入到 语料词序列文件中
                    fout.write(" \n")

                    doc_len_lst.append(len(lines_str.split()))

        print("Average length:", np.mean(doc_len_lst))
        print("doc count:", len(doc_len_lst))
        print("Total number of words:", len(word_st))


if __name__ == "__main__":
    CorpusProcess("logs")
