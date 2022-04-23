# TEXT_GCN /TEXT_AM-GCN  base on logs

## UseAag
**1. unzip datasets**

```
conda create -n python37 python=3.7
conda activate python37
cd .\PyTorch_TextGCN\
# cpu
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install scipy networkx tqdm scikit_learn pandas numpy spacy nltk prettytable fastai cloudpickle rocksdict -i https://pypi.tuna.tsinghua.edu.cn/simple

# 具体运行再main.py
```


## TIPS
每次运行前，如果想要清除数据，需要删除temp文件夹下面的cache文件夹


