# TEXT_GCN base on 20NG

20NG 数据集(bydate版本)包含18,846个文档，平均分为20个不同的类别。总共有11,314个文档在训练集中，7,532个文档在测试集中


## UseAag
```
conda create -n python37 python=3.7
conda activate python37
cd .\PyTorch_TextGCN\
# cpu
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install scipy networkx tqdm scikit_learn pandas numpy spacy nltk prettytable fastai cloudpickle rocksdict -i https://pypi.tuna.tsinghua.edu.cn/simple

python -c "import corpus_process;corpus_process.CorpusProcess('20ng')"
python -c "import graph_build_process;graph_build_process.GraphBuildProcess('20ng')"
python -c "import train_process,model;train_process.TrainProcess('20ng', 1,model.GCN,max_epoch=20).train()"
```


## model.py            ---模型
两层的GCN

>   参考：https://mp.weixin.qq.com/s/CpDZEqo14X_lCBh6i7feIA
> 
>   一个两层的GCN可以允许信息在最多两阶节点之间传递。因此，尽管图中文档和文档没有直接的边，但是两层的GCN允许文档对之间的信息交换。在我们的初步实验中。我们发现，双层GCN的性能优于单层GCN，而多层GCN的性能并没有提高。这类似于(Kipf and Welling 2017) 和 (Li, Han, and Wu 2018)的结果。



## corpus_process.py   ---语料处理
主要工作就是建立词序列，并写入到预处理结果文件中


>   我们首先通过清理和标记文本(Kim 2014)对所有数据集进行预处理。
    然后，我们删除了NLTK定义的停用词和20NG、R8、R52和Ohsumed中出现少于5次的低频字。
    唯一的例外是MR，我们在清理和标记原始文本后没有删除单词，因为文档非常短。(MR文本长度为250以下)
    


处理前：
```
From: decay@cbnewsj.cb.att.com (dean.kaflowitz) Subject: Re: about the bible quiz answers Organization: AT&T Distribution: na Lines: 18  In article <healta.153.735242337@saturn.wwc.edu>, healta@saturn.wwc.edu (Tammy R Healy) writes: >  >  > #12) The 2 cheribums are on the Ark of the Covenant.  When God said make no  > graven image, he was refering to idols, which were created to be worshipped.  > The Ark of the Covenant wasn't wrodhipped and only the high priest could  > enter the Holy of Holies where it was kept once a year, on the Day of  > Atonement.  I am not familiar with, or knowledgeable about the original language, but I believe there is a word for "idol" and that the translator would have used the word "idol" instead of "graven image" had the original said "idol."  So I think you're wrong here, but then again I could be too.  I just suggesting a way to determine whether the interpretation you offer is correct.   Dean Kaflowitz 

```


处理后：
```
decay cbnewsj cb att com \( dean kaflowitz \) subject bible quiz answers organization distribution na lines 18 article healta 153 saturn wwc edu , healta saturn wwc edu \( tammy r healy \) writes 12 \) 2 ark covenant god said make image , refering idols , created worshipped ark covenant n't high priest could enter holy kept year , day atonement familiar , knowledgeable original language , believe word idol translator would used word idol instead image original said idol think 're wrong , could suggesting way determine whether interpretation offer correct dean kaflowitz 

```

## graph_build_process.py      ---图构建
主要包含三部分
1. document-word tfidf edge 构建
2. word-word pmi edge 构建
3. networkx.Graph 的保存

>   我们简单地将特征矩阵 X = I  设置为一个单位矩阵，这意味着每个单词或文档都表示为One-Hot向量作为TextGCN的输入
>   
>   我们构建了一个包含单词节点和文档节点的大型异构文本图
> 
>   基于文档中单词共现(document-word edges)和整个语料库中单词共现(word-word edges)在节点之间构建边
> 
>   文档节点之间的边的重量和一个单词节点是文档中这个词的词频-逆文档频率(TF-IDF)，其中词频是在文档中单词出现的次数，逆文档频率是包含这个词的文档数量的倒数的对数。我们发现使用TF-IDF权重比只使用词频更好
> 
>   为了利用全局词的共现信息，我们对语料库中的所有文档使用一个固定大小的滑动窗口来收集共现统计信息。我们使用点互信息(PMI)，一种测量单词关联性的流行方法，来计算两个单词节点之间的权重
> 
>   ![节点i与节点j之间的边的权值定义为](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw05kgs9gwSa6B945EzyYmWTEI4xT0ZUhrb99VqZoES23HryPw6k69jc17jpiaLIkLvl4mqntrhDuew/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)
>
>   ![一对单词i, j的PMI值的计算公式为](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw05kgs9gwSa6B945EzyYmWT4o7PQQYOibia5BCC7azt0RakTNM4oeiaYMepBb3dJOZSYFS3MpuDm0xow/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)




## train_process.py            ---训练



## temp                ---存放中间文件
clean_corpus : 清理后的语料
graph：构建的图向量

## utils.py            ---工具函数


## datasets            ---存放数据
> describe           ---数据描述,同时也是target
> 
> 20ng corpus describe ,Each row corresponds to that

> corpus             ---原始的语料数据
> 
> 20ng corpus dataset


# Notes

## 20ng 语料仓库
https://github.com/poojahira/20-newsgroups