基于 **AM-GCN**的文本数据分类(text cluster base on AM-GCN)

that job you can ask `TEXT-AM-GCN`

that job base on `GCN`,`TEXT-GCN`,`AM-GCN`

that project just a stu demo

src:project source code
pre:pre stu source code
reference:reference doc


```bash
vocab_lst len: 1144
build document-word edges with tfidf: 2000it [00:00, 10478.15it/s]
document-word edges build time: 0.26105520000000015
+---------+---------+-------------+------------+----------+
|  Nodes  |  Edges  |  Selfloops  |  Isolates  |  覆盖度  |
+---------+---------+-------------+------------+----------+
|   3144  |  87033  |      0      |     0      |  1.0000  |
+---------+---------+-------------+------------+----------+
pmi read file len: 2000
build window: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [00:02<00:00, 825.03it/s] 
Calculate pmi between words: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 50764/50764 [00:00<00:00, 528794.53it/s] 
Total number of edges between word: 33318
get word-word edges build time 2.6001790999999996
+---------+----------+-------------+------------+----------+
|  Nodes  |  Edges   |  Selfloops  |  Isolates  |  覆盖度  |
+---------+----------+-------------+------------+----------+
|   3144  |  117016  |      0      |     0      |  1.0000  |
+---------+----------+-------------+------------+----------+
waiting fot write weighted_edgelist
GraphBuild success
get_features_tensor...: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [00:01<00:00, 1466.65it/s]
==> waiting for build knn graph data:logs, knn_graph_data_path: c:\Users\Administrator\Desktop\log_gcn\src/temp/knn/logs.txt <==
writing knn argpartition: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [00:00<00:00, 41099.07it/s]
writing knn data: 2000it [00:00, 208422.98it/s]
key:logs_graph is not in caches
==> waiting for read dataset graph_data:logs, graph_data_path: c:\Users\Administrator\Desktop\log_gcn\src/temp/graph/logs.txt <==
+---------+----------+-------------+------------+----------+
|  Nodes  |  Edges   |  Selfloops  |  Isolates  |  覆盖度  |
+---------+----------+-------------+------------+----------+
|   3144  |  117016  |      0      |     0      |  1.0000  |
+---------+----------+-------------+------------+----------+
key:logs_adjacency_matrix is not in caches
==> waiting for graph_data to scipy_sparse_matrix to torch_sparse_tensor <==
key:logs_knn_adjacency_matrix is not in caches
==> waiting for knn_graph to scipy_sparse_matrix to torch_sparse_tensor <==
key:logs_features is not in caches


==>Training 0, seed:40905  ...
{'lr': 0.02, 'device': device(type='cpu'), 'dataset': 'logs', 'max_epoch': 20, 'earlystopping': <utils.EarlyStopping object at 0x0000024613A70BC8>, 'nhid': 256, 'model': <class 'model.SFGCN'>, 'nclass': 2, 'dropout': 0.5, 'graph_nodes': 3144}
adjacency_matrix torch.Size([3144, 3144])
knn_adjacency_matrix torch.Size([3144, 3144])
features torch.Size([3144, 3144])
==> Start Train ... 
 <bound method Module.parameters of SFGCN(
  (SGCN1): GCN(
    (gc1): GraphConvolution (3144 -> 768)
    (gc2): GraphConvolution (768 -> 256)
  )
  (SGCN2): GCN(
    (gc1): GraphConvolution (3144 -> 768)
    (gc2): GraphConvolution (768 -> 256)
  )
  (CGCN): GCN(
    (gc1): GraphConvolution (3144 -> 768)
    (gc2): GraphConvolution (768 -> 256)
  )
  (attention): Attention(
    (project): Sequential(
      (0): Linear(in_features=256, out_features=16, bias=True)
      (1): Tanh()
      (2): Linear(in_features=16, out_features=1, bias=False)
    )
  )
  (tanh): Tanh()
  (MLP): Sequential(
    (0): Linear(in_features=256, out_features=2, bias=True)
    (1): LogSoftmax(dim=1)
  )
)>
epoch:16 train_loss:0.1444 valid_loss:0.5841 acc:0.9056 macro_f1:0.7921 precision:0.7974 recall:0.7869 :  80%|████████████████████████████▊       | 16/20 [00:56<00:14,  3.52s/it] 

```
