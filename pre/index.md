### 1.GCN

参考：

https://zhuanlan.zhihu.com/p/390836594

https://blog.csdn.net/d179212934/article/details/108093614

代码：

https://github.com/tkipf/pygcn

运行：

```
conda create -n python36 python=3.6
conda activate python36
cd .\pygcn\
python setup.py install
python train.py
```

输出：

```
Epoch: 0195 loss_train: 0.4486 acc_train: 0.9429 loss_val: 0.7045 acc_val: 0.8167 time: 0.0080s
Epoch: 0196 loss_train: 0.4215 acc_train: 0.9286 loss_val: 0.7034 acc_val: 0.8133 time: 0.0070s
Epoch: 0197 loss_train: 0.4695 acc_train: 0.9071 loss_val: 0.7030 acc_val: 0.8133 time: 0.0090s
Epoch: 0198 loss_train: 0.4331 acc_train: 0.9357 loss_val: 0.7031 acc_val: 0.8167 time: 0.0080s
Epoch: 0199 loss_train: 0.4246 acc_train: 0.9286 loss_val: 0.7033 acc_val: 0.8100 time: 0.0080s
Epoch: 0200 loss_train: 0.4651 acc_train: 0.9571 loss_val: 0.7027 acc_val: 0.8133 time: 0.0090s
Optimization Finished!
Total time elapsed: 1.9067s
Test set results: loss= 0.7570 accuracy= 0.8320
```

### 2.TEXT_GCN

参考：

https://zhuanlan.zhihu.com/p/56879815

https://mp.weixin.qq.com/s/CpDZEqo14X_lCBh6i7feIA


代码：

https://github.com/chengsen/PyTorch_TextGCN

运行：

```
conda create -n python37 python=3.7
conda activate python37
cd .\PyTorch_TextGCN\
# cpu
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple
# gpu
pip install scipy networkx tqdm scikit_learn pandas numpy spacy nltk prettytable fastai  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple

python -c "import data_processor;data_processor.CorpusProcess('20ng')"
python -c "import build_graph;build_graph.BuildGraph('20ng')"
python -c "import trainer;trainer.main('20ng', 1)"
```

输出：
```
epoch:78 train_loss:0.1089 val_loss:0.3358 acc:0.9214 macro_f1:0.9197 precision:0.9204 recall:0.9190
epoch:79 train_loss:0.1050 val_loss:0.3336 acc:0.9205 macro_f1:0.9189 precision:0.9197 recall:0.9182
epoch:80 train_loss:0.1039 val_loss:0.3341 acc:0.9205 macro_f1:0.9189 precision:0.9197 recall:0.9182
epoch:81 train_loss:0.1017 val_loss:0.3355 acc:0.9205 macro_f1:0.9193 precision:0.9204 recall:0.9182
0.8560
0.8560
0.8560
train_time:
1108.8063
1108.8063
1108.8063
model_param:
12324820.0000
12324820.0000
12324820.0000
```


### 3.理解 TEXT_GCN

参考：

https://blog.csdn.net/qq_28969139/article/details/105212185

https://github.com/iworldtong/text_gcn.pytorch

https://github.com/chengsen/PyTorch_TextGCN


代码：

位于`3_understand_text_gcn/text_gcn_demo`下

运行：
详见`3_understand_text_gcn/text_gcn_demo/readme.md`

### 4.AGCN

参考：

https://cloud.tencent.com/developer/article/1688778

https://github.com/zhumeiqiBUPT/AM-GCN

主要是理解 knn feature 在 AM-GCN 网络结构中的使用

